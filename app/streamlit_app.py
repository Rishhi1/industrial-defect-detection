"""
streamlit_app.py
=================
Phase 10: the full Streamlit web application tying every previous phase
together into an interactive industrial visual inspection dashboard.

Pages: Home, Inspection, Dataset, Model, Performance, About Project.

Run with:
    streamlit run app/streamlit_app.py

The app loads SAVED artifacts (memory banks, thresholds, evaluation
metrics/plots) produced by the CLI scripts in scripts/ — it never rebuilds
a memory bank or reruns evaluation itself. If those artifacts don't exist
yet for a category, the relevant page explains which script to run first,
rather than silently failing or attempting the work inline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.anomaly_detector import AnomalyDetector, MemoryBank
from src.dataset import SUPPORTED_EXTENSIONS, MVTecDatasetError, MVTecTestDataset, MVTecTrainDataset
from src.feature_extractor import FeatureExtractor
from src.localization import localize_anomaly
from src.preprocessing import build_inference_transform, denormalize
from src.utils import discover_categories, get_device, load_config

# ---------------------------------------------------------------------------
# Page config + light theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Industrial Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained industrial palette: graphite background panels, a warning-tape
# amber for "inspect this" accents, and a steady safety-green / safety-red
# pairing for NORMAL/ANOMALOUS — the same visual language as a real QC line
# indicator light, not a generic dashboard blue/purple gradient.
st.markdown(
    """
    <style>
    :root {
        --graphite: #2b2f36;
        --amber: #e8a33d;
        --safe-green: #2e7d32;
        --alert-red: #c62828;
    }
    .result-banner-normal {
        background-color: #e8f5e9;
        border: 2px solid var(--safe-green);
        color: var(--safe-green);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .result-banner-anomalous {
        background-color: #fdecea;
        border: 2px solid var(--alert-red);
        color: var(--alert-red);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .pipeline-stage {
        background-color: #f2f2f0;
        border: 1px solid #d8d8d5;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.92rem;
        color: #1a1a1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resource loaders — heavy objects loaded once, reused across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading feature extractor backbone...")
def get_feature_extractor(_config: dict, backbone_name: str) -> FeatureExtractor:
    # backbone_name is passed separately (hashable) purely so Streamlit's
    # cache key changes if the backbone changes; _config itself is unhashable
    # (dict) so the leading underscore tells st.cache_resource to skip hashing it.
    device = get_device(_config["model"]["device"])
    extractor = FeatureExtractor(_config).to(device)
    return extractor


@st.cache_resource(show_spinner=False)
def get_memory_bank_and_threshold(category: str, models_dir_str: str):
    models_dir = Path(models_dir_str)
    memory_bank_path = models_dir / f"{category}_memory_bank.pt"
    threshold_path = models_dir / f"{category}_threshold.json"

    if not memory_bank_path.exists() or not threshold_path.exists():
        return None, None

    memory_bank = MemoryBank.load(memory_bank_path)
    with open(threshold_path) as f:
        threshold_data = json.load(f)
    return memory_bank, threshold_data


@st.cache_data(show_spinner=False)
def get_dataset_stats(dataset_root_str: str, category: str) -> dict | None:
    """Lightweight dataset stats (file counts only — doesn't load pixel data)."""
    dataset_root = Path(dataset_root_str)
    try:
        train_ds = MVTecTrainDataset(dataset_root, category)
        test_ds = MVTecTestDataset(dataset_root, category)
    except MVTecDatasetError:
        return None

    breakdown = test_ds.summary()
    return {
        "train_normal": len(train_ds),
        "test_total": len(test_ds),
        "test_normal": breakdown.get("good", 0),
        "test_anomalous": len(test_ds) - breakdown.get("good", 0),
        "defect_types": {k: v for k, v in breakdown.items() if k != "good"},
    }


@st.cache_data(show_spinner=False)
def load_category_metrics(metrics_dir_str: str, category: str) -> dict | None:
    metrics_path = Path(metrics_dir_str) / f"{category}_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def usable_categories(config: dict) -> list[str]:
    """Categories that have BOTH a memory bank and a threshold — ready for inspection."""
    models_dir = Path(config["output"]["models_dir"])
    dataset_root = Path(config["dataset"]["root_path"])
    discovered = discover_categories(dataset_root) if dataset_root.exists() else []
    ready = []
    for cat in discovered:
        if (models_dir / f"{cat}_memory_bank.pt").exists() and (models_dir / f"{cat}_threshold.json").exists():
            ready.append(cat)
    return ready


# ---------------------------------------------------------------------------
# Page: HOME
# ---------------------------------------------------------------------------
def render_home(config: dict) -> None:
    st.title("AI-Based Industrial Quality Inspection System")
    st.subheader("Deep Learning Powered Anomaly Detection using MVTec AD")

    st.markdown(
        """
        This system inspects photos of industrial products and materials and flags
        whether each one is **NORMAL** or **ANOMALOUS**, then shows *where* the
        suspected defect is with a visual heatmap — a simplified version of an
        automated visual inspection line used in real manufacturing quality control.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### How it works")
        st.markdown(
            "- A frozen, pretrained CNN extracts features from the image\n"
            "- Features are compared against a reference bank built from normal images\n"
            "- The distance to the nearest normal patterns becomes the anomaly score\n"
            "- Scores above a data-driven threshold are flagged ANOMALOUS"
        )
    with col2:
        st.markdown("#### Technology used")
        st.markdown(
            f"- **Backbone:** {config['model']['backbone']}\n"
            f"- **Framework:** PyTorch + torchvision\n"
            f"- **Interface:** Streamlit\n"
            f"- **Method:** PaDiM/PatchCore-style feature memory bank"
        )
    with col3:
        st.markdown("#### Supported categories")
        ready = usable_categories(config)
        if ready:
            for cat in ready:
                st.markdown(f"- {cat.capitalize()} ✅")
        else:
            st.markdown("No categories are ready for inspection yet.")
            st.caption("Run the build_memory_bank.py and select_threshold.py scripts first.")

    st.divider()
    st.info(
        "Head to the **Inspection** page in the sidebar to upload an image and run "
        "a live defect check, or explore **Dataset**, **Model**, and **Performance** "
        "for more detail on how the system works and how well it performs."
    )


# ---------------------------------------------------------------------------
# Page: INSPECTION
# ---------------------------------------------------------------------------
def render_inspection(config: dict) -> None:
    st.title("Inspection")
    st.caption("Upload an image and run a live anomaly check against the trained reference model.")

    ready = usable_categories(config)
    if not ready:
        st.error(
            "No categories are ready for inspection. A category needs both a memory bank "
            "and a selected threshold before it can be used here. Run:\n\n"
            "```\npython scripts/build_memory_bank.py\npython scripts/select_threshold.py\n```"
        )
        return

    col_left, col_right = st.columns([1, 1])
    with col_left:
        category = st.selectbox("Select industrial category", ready)
    with col_right:
        uploaded_file = st.file_uploader(
            "Upload an image", type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]
        )

    if uploaded_file is None:
        st.caption("Upload an image above to run inspection.")
        return

    try:
        raw_image = Image.open(uploaded_file)
        raw_image.load()
        raw_image = raw_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        st.error(f"Could not read the uploaded file — it may be corrupted or not a valid image: {e}")
        return

    run_button = st.button("Run Inspection", type="primary")
    if not run_button:
        st.image(raw_image, caption="Uploaded image (not yet analyzed)", width=350)
        return

    memory_bank, threshold_data = get_memory_bank_and_threshold(category, config["output"]["models_dir"])
    if memory_bank is None:
        st.error(f"No memory bank/threshold found for '{category}'. Run the build scripts first.")
        return

    extractor = get_feature_extractor(config, config["model"]["backbone"])
    detector = AnomalyDetector(extractor, memory_bank, config)
    threshold = threshold_data["threshold"]

    with st.spinner("Running inference..."):
        start_time = time.time()

        transform = build_inference_transform(config)
        image_tensor = transform(raw_image).unsqueeze(0)
        device = get_device(config["model"]["device"])
        image_tensor = image_tensor.to(device)

        score_map = detector.compute_patch_score_map(image_tensor)[0]
        image_score = float(detector.compute_image_scores(image_tensor)[0])
        prediction = "ANOMALOUS" if image_score >= threshold else "NORMAL"

        image_size = config["preprocessing"]["image_size"]
        original_rgb = denormalize(image_tensor[0].cpu(), config)
        result = localize_anomaly(
            score_map, original_rgb, image_size, threshold=threshold, normalization_mode="threshold_relative"
        )

        processing_time = time.time() - start_time

    # --- Result banner ---
    banner_class = "result-banner-normal" if prediction == "NORMAL" else "result-banner-anomalous"
    icon = "✅" if prediction == "NORMAL" else "⚠️"
    st.markdown(f'<div class="{banner_class}">{icon} Inspection Result: {prediction}</div>', unsafe_allow_html=True)
    st.write("")

    # --- Metric cards ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomaly Score", f"{image_score:.3f}")
    m2.metric("Threshold", f"{threshold:.3f}")
    m3.metric("Margin", f"{image_score - threshold:+.3f}")
    m4.metric("Processing Time", f"{processing_time:.3f}s")

    st.caption(
        "Anomaly Score is a distance-based measure — how far this image's patches sit from "
        "anything seen during normal-image training — not a calibrated probability."
    )

    # --- Image comparison ---
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(original_rgb, caption="Original", width='stretch')
    with c2:
        st.image(result["heatmap_colored"], caption="Anomaly Heatmap", width='stretch')
    with c3:
        st.image(result["overlay"], caption="Overlay", width='stretch')


# ---------------------------------------------------------------------------
# Page: DATASET
# ---------------------------------------------------------------------------
def render_dataset(config: dict) -> None:
    st.title("Dataset")
    st.caption("Overview of the MVTec AD dataset categories used by this system.")

    st.markdown(
        """
        **MVTec AD** (MVTec Anomaly Detection) is a dataset for benchmarking anomaly
        detection methods on industrial inspection tasks. Each category ships a
        **train** split containing only defect-free ("good") images, and a **test**
        split containing both defect-free images and several distinct defect types,
        each with a pixel-level ground-truth mask.
        """
    )

    dataset_root = Path(config["dataset"]["root_path"])
    if not dataset_root.exists():
        st.warning(f"Dataset root '{dataset_root}' not found. Download MVTec AD and place it there.")
        return

    discovered = discover_categories(dataset_root)
    if not discovered:
        st.warning(f"No valid category folders found under '{dataset_root}'.")
        return

    for category in discovered:
        stats = get_dataset_stats(str(dataset_root), category)
        if stats is None:
            continue

        with st.expander(f"**{category.capitalize()}**", expanded=(category == discovered[0])):
            c1, c2, c3 = st.columns(3)
            c1.metric("Training images (normal)", stats["train_normal"])
            c2.metric("Test images (normal)", stats["test_normal"])
            c3.metric("Test images (anomalous)", stats["test_anomalous"])

            if stats["defect_types"]:
                st.markdown("**Defect types in test set:**")
                defect_df = pd.DataFrame(
                    [{"Defect Type": k, "Count": v} for k, v in stats["defect_types"].items()]
                )
                st.dataframe(defect_df, hide_index=True, width='stretch')

            # Sample images: a couple of normal + a couple of defective, if quick to load.
            try:
                train_ds = MVTecTrainDataset(dataset_root, category)
                sample_paths = train_ds.image_paths[:3]
                if sample_paths:
                    st.markdown("**Sample normal training images:**")
                    cols = st.columns(len(sample_paths))
                    for col, path in zip(cols, sample_paths):
                        col.image(str(path), width='stretch')
            except MVTecDatasetError:
                pass


# ---------------------------------------------------------------------------
# Page: MODEL
# ---------------------------------------------------------------------------
def render_model(config: dict) -> None:
    st.title("Model")
    st.caption("How the anomaly detection pipeline works, end to end.")

    st.markdown("### Pipeline")
    stages = [
        "Input Image",
        "Preprocessing (resize, normalize)",
        f"Pretrained {config['model']['backbone']} — feature extraction ({', '.join(config['model']['feature_layers'])})",
        "Patch-level feature embeddings",
        "Compare against Normal Reference Memory Bank (k-NN distance)",
        "Anomaly Score per patch",
        f"Threshold ({config['threshold']['method']}) → NORMAL / ANOMALOUS",
        "Upsample patch scores → Anomaly Heatmap",
        "Final Result: label + score + heatmap overlay",
    ]
    for i, stage in enumerate(stages):
        arrow = " ↓" if i < len(stages) - 1 else ""
        st.markdown(f'<div class="pipeline-stage">{i+1}. {stage}</div>{arrow}', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Extraction")
        st.markdown(
            f"""
            A frozen, ImageNet-pretrained **{config['model']['backbone']}** acts as a
            fixed feature extractor — no fine-tuning, no gradient updates. Intermediate
            layers **{', '.join(config['model']['feature_layers'])}** are tapped via
            forward hooks (not the final classification layer), since mid-level layers
            balance semantic richness with spatial resolution needed for localization.

            Feature maps from each configured layer are upsampled to a common
            resolution and concatenated channel-wise into one patch-embedding grid —
            each spatial location gets one combined feature vector.
            """
        )

    with col2:
        st.markdown("### Reference Memory Bank")
        st.markdown(
            f"""
            Every **normal** ("good") training image is run through the feature
            extractor. All resulting patch vectors are pooled and subsampled
            (coreset ratio: **{config['memory_bank']['coreset_ratio']}**, method:
            **{config['memory_bank']['coreset_method']}**) to build a compact
            reference set — this is what "normal" looks like in feature space.

            No training loop, no backpropagation: building the memory bank is
            purely forward passes plus storage.
            """
        )

    st.markdown("### Anomaly Scoring & Localization")
    st.markdown(
        f"""
        At inference, each patch of a new image is compared to its
        **{config['memory_bank']['k_nearest_neighbors']} nearest neighbors** in the
        memory bank (Euclidean distance) — patches unlike anything seen in normal
        training score higher. The image-level score is the
        **{config['threshold']['aggregation']}** of all patch scores in that image.

        For localization, the full grid of patch scores is upsampled back to the
        original image resolution and rendered as a color heatmap overlay, showing
        which regions drove the anomaly decision.
        """
    )


# ---------------------------------------------------------------------------
# Page: PERFORMANCE
# ---------------------------------------------------------------------------
def render_performance(config: dict) -> None:
    st.title("Performance")
    st.caption("Evaluation results from the last full run of scripts/evaluate.py.")

    metrics_dir = Path(config["output"]["metrics_dir"])
    plots_dir = Path(config["output"]["plots_dir"])
    summary_path = metrics_dir / "evaluation_summary.json"

    if not summary_path.exists():
        st.warning(
            "No evaluation results found. Run:\n\n```\npython scripts/evaluate.py\n```\n\n"
            "to generate metrics and plots first."
        )
        return

    with open(summary_path) as f:
        summary = json.load(f)

    categories = list(summary["per_category"].keys())
    if not categories:
        st.warning("Evaluation summary exists but contains no categories.")
        return

    st.markdown("### Category-Wise Results")
    rows = []
    for cat, m in summary["per_category"].items():
        rows.append({
            "Category": cat,
            "Images": m["num_images"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Pixel ROC-AUC": round(m["pixel_roc_auc"], 4) if m.get("pixel_roc_auc") is not None else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    pooled = summary.get("pooled", {})
    macro = summary.get("macro", {})
    st.markdown("### Overall Metrics")
    st.caption(
        "Two aggregation methods are shown since they answer different questions: "
        "**Pooled** combines every test image across categories (larger categories "
        "count proportionally more). **Macro** averages each category's metric with "
        "equal weight, regardless of test set size."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pooled** ({pooled.get('num_images', '?')} images)")
        st.metric("Accuracy", f"{pooled.get('accuracy', float('nan')):.4f}")
        st.metric("F1", f"{pooled.get('f1', float('nan')):.4f}")
    with c2:
        st.markdown(f"**Macro** ({macro.get('num_categories', '?')} categories)")
        st.metric("Accuracy", f"{macro.get('accuracy', float('nan')):.4f}")
        st.metric("ROC-AUC", f"{macro.get('roc_auc', float('nan')):.4f}")

    st.divider()
    st.markdown("### Category Comparison Charts")
    chart_cols = st.columns(2)
    roc_chart = plots_dir / "category_comparison_roc_auc.png"
    f1_chart = plots_dir / "category_comparison_f1.png"
    if roc_chart.exists():
        chart_cols[0].image(str(roc_chart), caption="ROC-AUC by Category", width='stretch')
    if f1_chart.exists():
        chart_cols[1].image(str(f1_chart), caption="F1 Score by Category", width='stretch')

    st.divider()
    st.markdown("### Per-Category Detail")
    selected_cat = st.selectbox("Select a category to inspect in detail", categories)
    cat_plots_dir = plots_dir / selected_cat

    plot_files = {
        "ROC Curve": cat_plots_dir / "roc_curve.png",
        "Precision-Recall Curve": cat_plots_dir / "pr_curve.png",
        "Confusion Matrix": cat_plots_dir / "confusion_matrix.png",
        "Score Distribution": cat_plots_dir / "score_distribution.png",
    }
    existing_plots = {name: path for name, path in plot_files.items() if path.exists()}

    if not existing_plots:
        st.info(f"No saved plots found for '{selected_cat}' under '{cat_plots_dir}'.")
    else:
        plot_cols = st.columns(2)
        for i, (name, path) in enumerate(existing_plots.items()):
            plot_cols[i % 2].image(str(path), caption=name, width='stretch')


# ---------------------------------------------------------------------------
# Page: ABOUT PROJECT
# ---------------------------------------------------------------------------
def render_about(config: dict) -> None:
    st.title("About This Project")

    st.markdown(
        """
        **Project Title:** Deep Learning Based Industrial Surface Defect Detection
        and Anomaly Inspection System Using MVTec AD Dataset

        **Student Name:** _[fill in]_
        **Roll Number:** _[fill in]_
        **College:** _[fill in]_
        **Department:** _[fill in]_
        **Training Organization:** _[fill in]_
        **Training Duration:** _[fill in]_
        """
    )

    st.divider()
    st.markdown("### Technologies Used")
    st.markdown(
        "- Python, PyTorch, torchvision\n"
        "- NumPy, Pandas, OpenCV, Pillow, scikit-learn\n"
        "- Matplotlib, Seaborn\n"
        "- Streamlit\n"
        f"- Backbone: {config['model']['backbone']} (ImageNet-pretrained)"
    )

    st.markdown("### Future Scope")
    st.markdown(
        "- Extend to the full MVTec AD category set\n"
        "- Explore PatchCore-style greedy coreset selection at larger scale\n"
        "- Add per-defect-type breakdown to the evaluation suite\n"
        "- Package as a lightweight on-device inspection tool for edge deployment"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        config = load_config("configs/config.yaml")
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Failed to load configuration: {e}")
        st.stop()

    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Inspection", "Dataset", "Model", "Performance", "About Project"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    device = get_device(config["model"]["device"])
    st.sidebar.caption(f"Device: {device}")
    st.sidebar.caption(f"Backbone: {config['model']['backbone']}")

    pages = {
        "Home": render_home,
        "Inspection": render_inspection,
        "Dataset": render_dataset,
        "Model": render_model,
        "Performance": render_performance,
        "About Project": render_about,
    }
    pages[page](config)


if __name__ == "__main__":
    main()
