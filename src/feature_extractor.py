"""
feature_extractor.py
=====================
Wraps a pretrained, frozen torchvision CNN backbone (ResNet18 or ResNet50)
and extracts intermediate patch-level feature maps via forward hooks,
instead of the final classification output.

Why intermediate layers, not the final layer:
- Early layers (conv1, layer1) are too generic/low-level (edges, colors).
- The final layer/avgpool collapses all spatial information — we need the
  spatial grid intact to localize *where* an anomaly is, not just whether
  one exists.
- Mid-level layers (layer2, layer3 in ResNet) are the standard choice in
  PaDiM/PatchCore-style methods: rich enough semantically to distinguish
  normal vs. defective texture/structure, while still preserving a spatial
  grid fine enough for localization.

The backbone is used purely as a fixed feature extractor: no gradients, no
fine-tuning, always in eval() mode. This is why memory-bank construction
takes minutes, not hours — there's no training loop at all, just forward
passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Layers we know how to attach hooks to for each supported backbone, and the
# number of output channels each one produces (needed later for memory bank
# sizing / documentation, not strictly required for the hook mechanism itself).
_RESNET_LAYER_CHANNELS = {
    "layer1": 64,   # resnet18/34; resnet50/101 use 256 (bottleneck expansion)
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
}
_RESNET50_LAYER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


def _build_backbone(name: str) -> nn.Module:
    """
    Load an ImageNet-pretrained torchvision backbone with its classification
    head removed (we never use it — only intermediate feature maps).
    """
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(
            f"Unsupported backbone '{name}'. Supported: 'resnet18', 'resnet50'. "
            f"(Set model.backbone in config.yaml.)"
        )

    # Freeze everything — this is a fixed feature extractor, never fine-tuned.
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


def layer_channels(backbone_name: str, layer_name: str) -> int:
    """Return the number of output channels for a given backbone/layer combo."""
    table = _RESNET50_LAYER_CHANNELS if backbone_name.lower() == "resnet50" else _RESNET_LAYER_CHANNELS
    if layer_name not in table:
        raise ValueError(f"Unknown layer '{layer_name}' for backbone '{backbone_name}'.")
    return table[layer_name]


class FeatureExtractor:
    """
    Extracts intermediate feature maps from a frozen pretrained backbone
    using forward hooks, and combines them into a single spatially-aligned
    patch-embedding grid (the PaDiM-style "concatenate after upsampling to
    a common resolution" approach).

    Usage:
        extractor = FeatureExtractor(config)
        extractor.to(device)
        raw_features = extractor.extract(batch)           # dict of layer -> tensor
        patch_embed = extractor.get_patch_embeddings(batch)  # single tensor
    """

    def __init__(self, config: dict):
        self.backbone_name = config["model"]["backbone"]
        self.feature_layers: list[str] = config["model"]["feature_layers"]

        if len(self.feature_layers) == 0:
            raise ValueError("config.model.feature_layers must list at least one layer.")

        self.backbone = _build_backbone(self.backbone_name)
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    # -----------------------------------------------------------------
    def _register_hooks(self) -> None:
        """Attach a forward hook to each configured layer that stores its output."""
        for layer_name in self.feature_layers:
            if not hasattr(self.backbone, layer_name):
                raise ValueError(
                    f"Backbone '{self.backbone_name}' has no layer named '{layer_name}'. "
                    f"Valid ResNet layer names: layer1, layer2, layer3, layer4."
                )
            layer = getattr(self.backbone, layer_name)

            def make_hook(name):
                def hook(module, input, output):
                    self._activations[name] = output
                return hook

            handle = layer.register_forward_hook(make_hook(layer_name))
            self._hooks.append(handle)

    def to(self, device: torch.device) -> "FeatureExtractor":
        self.backbone = self.backbone.to(device)
        return self

    # -----------------------------------------------------------------
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Run a batch of preprocessed images (B, 3, H, W) through the backbone
        and return the raw intermediate feature maps captured by the hooks,
        one per configured layer: {"layer2": (B, C2, H2, W2), "layer3": (B, C3, H3, W3), ...}
        """
        self._activations = {}
        _ = self.backbone(images)  # forward pass; outputs discarded, hooks capture what we need

        missing = [l for l in self.feature_layers if l not in self._activations]
        if missing:
            raise RuntimeError(
                f"Forward hooks did not fire for layer(s) {missing}. This usually means "
                f"the backbone's forward() doesn't actually execute those layers — check "
                f"config.model.feature_layers."
            )

        return {layer: self._activations[layer] for layer in self.feature_layers}

    @torch.no_grad()
    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple layers' feature maps into a single spatially-aligned
        patch-embedding grid, following the PaDiM approach:

        1. Extract raw feature maps from each configured layer (different
           layers have different spatial resolutions — deeper layers are
           smaller).
        2. Upsample every layer's feature map to match the resolution of the
           FIRST (largest / shallowest) configured layer, using bilinear
           interpolation.
        3. Concatenate all layers along the channel dimension.

        Result: (B, C_total, H, W) where each spatial location (h, w) has an
        embedding vector describing that image patch, built from multiple
        levels of the network's semantic hierarchy. This patch grid is what
        gets stored in the memory bank and compared against at inference.
        """
        raw = self.extract(images)

        target_layer = self.feature_layers[0]
        target_h, target_w = raw[target_layer].shape[-2:]

        aligned = []
        for layer_name in self.feature_layers:
            feat = raw[layer_name]
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(feat)

        return torch.cat(aligned, dim=1)  # (B, C_total, target_h, target_w)

    def total_embedding_channels(self) -> int:
        """Total channel count of the combined patch embedding (sum across configured layers)."""
        return sum(layer_channels(self.backbone_name, layer) for layer in self.feature_layers)

    def __del__(self):
        # Clean up hooks if the extractor is garbage collected.
        for handle in getattr(self, "_hooks", []):
            handle.remove()
