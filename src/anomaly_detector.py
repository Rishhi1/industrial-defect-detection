"""
anomaly_detector.py
====================
Implements the core of the anomaly detection system: a reference "memory
bank" of patch-level feature embeddings from NORMAL training images, and
scoring logic that measures how far a new image's patches are from anything
seen in that reference set.

This is a PaDiM/PatchCore-style approach:

1. Run every normal training image through the feature extractor to get a
   patch-embedding grid (see feature_extractor.py).
2. Flatten all (image, spatial-location) pairs into one big pool of patch
   vectors: N images x (H x W) patches each -> (N*H*W, C) vectors.
3. Optionally subsample this pool ("coreset") to keep memory/compute
   manageable, since keeping every single patch from every training image
   doesn't scale well on a student laptop.
4. At inference, for a query image's patch grid, compute the distance from
   each query patch to its nearest neighbor(s) in the memory bank. Patches
   that look nothing like anything in the normal reference set get a high
   anomaly score.
5. Image-level anomaly score = the maximum (or top-k mean) patch score,
   since a single strongly anomalous region is enough to flag the whole
   image as defective.

No gradient descent happens anywhere in this file — building the memory
bank is purely forward passes + storage, which is why it takes minutes,
not hours, even on a CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from src.dataset import MVTecTrainDataset
from src.feature_extractor import FeatureExtractor
from src.preprocessing import build_inference_transform
from src.utils import ensure_dir, log_info


@dataclass
class MemoryBankMetadata:
    category: str
    backbone: str
    feature_layers: list[str]
    image_size: int
    num_training_images: int
    total_patches_before_coreset: int
    num_patches_after_coreset: int
    coreset_ratio: float
    coreset_method: str
    embedding_dim: int
    patch_grid_size: tuple[int, int]  # (H, W) of the patch embedding grid
    random_seed: int
    build_time_seconds: float = 0.0


class MemoryBank:
    """
    Holds the reference set of normal-image patch embeddings for ONE MVTec
    category, plus the metadata needed to rebuild a compatible feature
    extractor and interpret scores correctly.
    """

    def __init__(self, embeddings: np.ndarray, metadata: MemoryBankMetadata):
        self.embeddings = embeddings  # (M, C) float32
        self.metadata = metadata
        self._nn_index: NearestNeighbors | None = None

    # -----------------------------------------------------------------
    def _ensure_index(self, k: int) -> NearestNeighbors:
        """Lazily build (and cache) the k-NN index over the memory bank."""
        if self._nn_index is None:
            self._nn_index = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
            self._nn_index.fit(self.embeddings)
        return self._nn_index

    def query(self, patch_vectors: np.ndarray, k: int) -> np.ndarray:
        """
        For each query patch vector, return the mean distance to its k
        nearest neighbors in the memory bank. Higher = more anomalous
        (further from anything seen during normal-image training).

        patch_vectors: (Q, C)
        returns: (Q,) anomaly scores
        """
        index = self._ensure_index(k)
        distances, _ = index.kneighbors(patch_vectors, n_neighbors=k)
        return distances.mean(axis=1)

    # -----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "embeddings": self.embeddings,
                "metadata": self.metadata.__dict__,
            },
            path,
        )
        log_info(f"Saved memory bank ({self.embeddings.shape[0]} patches, {self.embeddings.shape[1]}-d) to '{path}'")

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Memory bank file not found at '{path}'. Run scripts/build_memory_bank.py first."
            )
        data = torch.load(path, weights_only=False)
        metadata = MemoryBankMetadata(**data["metadata"])
        return cls(embeddings=data["embeddings"], metadata=metadata)


# -----------------------------------------------------------------------
# Coreset subsampling
# -----------------------------------------------------------------------
def random_coreset(embeddings: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """
    Randomly subsample a fraction of patch embeddings. Fast (O(N)) and the
    default choice for demo mode — with enough patches even a random subset
    reasonably covers the normal-appearance distribution.
    """
    if ratio >= 1.0:
        return embeddings
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(len(embeddings) * ratio))
    indices = rng.choice(len(embeddings), size=n_keep, replace=False)
    return embeddings[indices]


def greedy_kcenter_coreset(embeddings: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """
    Greedy k-center coreset selection (PatchCore-style): iteratively picks
    the point farthest from every already-selected center, which tends to
    give better coverage of the embedding space than random sampling.

    Much slower than random_coreset — O(k * N) distance computations, where
    k is the target coreset size. Intended for "full mode" on a smaller
    patch pool, not as the default for large demo-mode runs. A progress
    log line is printed periodically since this can take a while.
    """
    if ratio >= 1.0:
        return embeddings

    n_total = len(embeddings)
    n_keep = max(1, int(n_total * ratio))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(embeddings).float().to(device)

    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, n_total))

    selected_indices = [first_idx]
    min_dist = torch.full((n_total,), float("inf"), device=device)

    log_info(f"Building greedy k-center coreset: selecting {n_keep} of {n_total} patches...")
    start = time.time()

    last_selected = points[first_idx].unsqueeze(0)
    for i in range(1, n_keep):
        dist_to_last = torch.cdist(points, last_selected).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_to_last)
        next_idx = int(torch.argmax(min_dist).item())
        selected_indices.append(next_idx)
        last_selected = points[next_idx].unsqueeze(0)

        if i % max(1, n_keep // 10) == 0:
            elapsed = time.time() - start
            log_info(f"  ... {i}/{n_keep} selected ({elapsed:.1f}s elapsed)")

    log_info(f"Greedy coreset selection finished in {time.time() - start:.1f}s")
    return embeddings[selected_indices]


CORESET_METHODS = {
    "random": random_coreset,
    "greedy_kcenter": greedy_kcenter_coreset,
}


# -----------------------------------------------------------------------
# Memory bank construction
# -----------------------------------------------------------------------
def build_memory_bank(
    dataset_root: str | Path,
    category: str,
    config: dict,
    extractor: FeatureExtractor,
    device: torch.device,
) -> MemoryBank:
    """
    Build a MemoryBank for one category from its normal ('good') training
    images: extract patch embeddings for every training image, flatten,
    and subsample down to the configured coreset ratio.
    """
    start_time = time.time()

    transform = build_inference_transform(config)
    train_ds = MVTecTrainDataset(dataset_root, category, transform=transform)

    batch_size = config["dataloader"]["batch_size"]
    num_workers = config["dataloader"]["num_workers"]
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    log_info(f"[{category}] Extracting patch embeddings from {len(train_ds)} normal training images...")

    all_patches = []
    patch_grid_size = None

    for batch in loader:
        batch = batch.to(device)
        patch_embed = extractor.get_patch_embeddings(batch)  # (B, C, H, W)
        b, c, h, w = patch_embed.shape
        patch_grid_size = (h, w)

        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C): one row per spatial patch
        flattened = patch_embed.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        all_patches.append(flattened)

    all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
    total_patches = all_patches.shape[0]
    embedding_dim = all_patches.shape[1]

    log_info(f"[{category}] Extracted {total_patches} total patches (dim={embedding_dim}).")

    coreset_ratio = config["memory_bank"]["coreset_ratio"]
    coreset_method_name = config["memory_bank"].get("coreset_method", "random")
    seed = config["memory_bank"]["random_seed"]

    if coreset_method_name not in CORESET_METHODS:
        raise ValueError(
            f"Unknown coreset_method '{coreset_method_name}'. "
            f"Valid options: {list(CORESET_METHODS.keys())}"
        )

    coreset_fn = CORESET_METHODS[coreset_method_name]
    log_info(f"[{category}] Applying '{coreset_method_name}' coreset subsampling (ratio={coreset_ratio})...")
    subsampled = coreset_fn(all_patches, coreset_ratio, seed)

    log_info(f"[{category}] Memory bank size after subsampling: {subsampled.shape[0]} patches "
              f"({subsampled.shape[0] / total_patches * 100:.1f}% of original).")

    metadata = MemoryBankMetadata(
        category=category,
        backbone=config["model"]["backbone"],
        feature_layers=list(config["model"]["feature_layers"]),
        image_size=config["preprocessing"]["image_size"],
        num_training_images=len(train_ds),
        total_patches_before_coreset=total_patches,
        num_patches_after_coreset=subsampled.shape[0],
        coreset_ratio=coreset_ratio,
        coreset_method=coreset_method_name,
        embedding_dim=embedding_dim,
        patch_grid_size=patch_grid_size,
        random_seed=seed,
        build_time_seconds=time.time() - start_time,
    )

    return MemoryBank(embeddings=subsampled, metadata=metadata)


# -----------------------------------------------------------------------
# Anomaly scoring — combines a FeatureExtractor with a MemoryBank to score
# new (test/inference) images against the normal reference set.
# -----------------------------------------------------------------------
class AnomalyDetector:
    """
    Scores images for anomalousness using a fitted FeatureExtractor +
    MemoryBank pair. This is the object every downstream script (threshold
    selection, evaluation, inference, the Streamlit app) uses to actually
    classify an image as NORMAL or ANOMALOUS.

    Two levels of output are provided:
    - compute_patch_score_map(): per-spatial-location anomaly scores
      (used for heatmap localization in Phase 7).
    - compute_image_scores(): a single scalar per image, aggregated from
      the patch score map (used for the NORMAL/ANOMALOUS decision).
    """

    def __init__(self, extractor: FeatureExtractor, memory_bank: MemoryBank, config: dict):
        self.extractor = extractor
        self.memory_bank = memory_bank
        self.k = config["memory_bank"]["k_nearest_neighbors"]
        self.aggregation = config["threshold"]["aggregation"]
        self.topk_mean_k = config["threshold"].get("topk_mean_k", 10)

    @torch.no_grad()
    def compute_patch_score_map(self, images: torch.Tensor) -> np.ndarray:
        """
        images: (B, 3, H, W) preprocessed batch.
        Returns: (B, h, w) anomaly score per spatial patch location, where
        (h, w) is the feature extractor's patch grid size (e.g. 28x28),
        NOT the original image resolution — upsampling to image size happens
        in the localization step (Phase 7).
        """
        patch_embed = self.extractor.get_patch_embeddings(images)  # (B, C, h, w)
        b, c, h, w = patch_embed.shape
        flattened = patch_embed.permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
        scores = self.memory_bank.query(flattened, k=self.k)  # (B*h*w,)
        return scores.reshape(b, h, w)

    def compute_image_scores(self, images: torch.Tensor) -> np.ndarray:
        """
        Returns: (B,) scalar anomaly score per image, aggregated from the
        patch score map according to config.threshold.aggregation.
        """
        score_map = self.compute_patch_score_map(images)  # (B, h, w)
        flat = score_map.reshape(score_map.shape[0], -1)  # (B, h*w)

        if self.aggregation == "max":
            return flat.max(axis=1)
        elif self.aggregation == "topk_mean":
            k = min(self.topk_mean_k, flat.shape[1])
            top_k = np.sort(flat, axis=1)[:, -k:]
            return top_k.mean(axis=1)
        else:
            raise ValueError(
                f"Unknown aggregation method '{self.aggregation}'. Valid: 'max', 'topk_mean'."
            )


@dataclass
class ThresholdResult:
    threshold: float
    method: str
    roc_auc: float
    tpr_at_threshold: float
    fpr_at_threshold: float
    num_normal: int
    num_anomalous: int
    aggregation: str


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> ThresholdResult:
    """
    Choose the NORMAL/ANOMALOUS cutoff from image-level anomaly scores and
    ground-truth labels (0=normal, 1=anomalous), using the method configured
    in config.threshold.method.

    "youden_j": the ROC-optimal threshold — the point that maximizes
        TPR - FPR (Youden's J statistic). Requires both classes present.
        This is computed on the TEST set, which is a known simplification
        (a fully rigorous setup would reserve a separate validation split)
        — acceptable and common for coursework-scale anomaly detection
        projects, but worth stating plainly rather than overclaiming rigor.

    "percentile": a fixed percentile of the NORMAL images' scores (e.g. the
        99th percentile) is used as the cutoff. Doesn't require anomalous
        examples at all, so it's the fallback when only normal scores are
        available or when youden_j can't be computed (e.g. only one class
        present in the provided data).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    method = config["threshold"]["method"]
    fallback_percentile = config["threshold"]["fallback_percentile"]
    aggregation = config["threshold"]["aggregation"]

    num_normal = int((labels == 0).sum())
    num_anomalous = int((labels == 1).sum())

    can_use_youden = method == "youden_j" and num_normal > 0 and num_anomalous > 0

    if can_use_youden:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        threshold = float(thresholds[best_idx])
        auc = float(roc_auc_score(labels, scores))

        return ThresholdResult(
            threshold=threshold,
            method="youden_j",
            roc_auc=auc,
            tpr_at_threshold=float(tpr[best_idx]),
            fpr_at_threshold=float(fpr[best_idx]),
            num_normal=num_normal,
            num_anomalous=num_anomalous,
            aggregation=aggregation,
        )

    # Fallback: percentile of normal scores. Also used if youden_j was
    # requested but the data doesn't support it (e.g. no anomalous samples).
    if method == "youden_j":
        from src.utils import log_warning
        log_warning(
            f"Requested method 'youden_j' but data has {num_normal} normal / "
            f"{num_anomalous} anomalous samples — need both classes present. "
            f"Falling back to 'percentile' method instead."
        )

    normal_scores = scores[labels == 0] if num_normal > 0 else scores
    threshold = float(np.percentile(normal_scores, fallback_percentile))

    # Report AUC too if both classes happen to be available, for reference.
    auc = float(roc_auc_score(labels, scores)) if (num_normal > 0 and num_anomalous > 0) else float("nan")
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tpr = tp / num_anomalous if num_anomalous > 0 else float("nan")
    fpr = fp / num_normal if num_normal > 0 else float("nan")

    return ThresholdResult(
        threshold=threshold,
        method="percentile",
        roc_auc=auc,
        tpr_at_threshold=tpr,
        fpr_at_threshold=fpr,
        num_normal=num_normal,
        num_anomalous=num_anomalous,
        aggregation=aggregation,
    )
