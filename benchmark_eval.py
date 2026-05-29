#!/usr/bin/env python3
"""
Benchmark evaluation for gLM2 DIAMOND pipeline.
Produces confusion matrices, ROC curves, and comparison figures.

Confusion matrix definitions (per Wikipedia):
  Positive class  = seqhub reference sequences  (P = 100)
  Negative class  = non-reference candidates    (N = candidates_embedded - P)
  Predicted positive at rank k = sequences in top-k
  Predicted negative at rank k = sequences not in top-k

Usage:
    python benchmark_eval.py
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REFERENCE_FASTA = "seqhub_matches.fa"  # ground truth from seqhub

PIPELINES = [

    {
        "label": "full-seq\n+ <+> prefix",
        "label_short": "full-seq+prefix",
        "file": "results_50k_fullseqs_prefixed.json",
        "color": "#2ca02c",
    },
]

OUT_DIR = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 101  # operating point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_id(sid: str) -> str:
    return sid.replace("|forward|", "|+|").replace("|reverse|", "|-|")


def load_reference(fasta_path: str) -> set[str]:
    """Return set of normalized seqhub reference IDs (excluding the query itself)."""
    ref = set()
    for rec in SeqIO.parse(fasta_path, "fasta"):
        nid = norm_id(rec.id)
        if nid.lower() != "query":
            ref.add(nid)
    return ref


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def evaluate(results_file: str, reference_ids: set[str], top_k: int = TOP_K) -> dict:
    """Compute full confusion-matrix metrics for one pipeline result file."""
    data = json.loads(Path(results_file).read_text())
    P     = len(reference_ids)
    total = data.get("candidates_embedded", data.get("diamond_hits", 50000))
    N     = total - P                    # non-reference candidates in pool

    matches = data["results"][0]["matches"][:top_k]

    retrieved_ids = {norm_id(m["corpus_id"]) for m in matches}
    tp = len(retrieved_ids & reference_ids)
    fp = len(retrieved_ids - reference_ids)
    fn = P - tp
    tn = N - fp

    # Standard metrics (Wikipedia confusion-matrix table)
    tpr  = tp / P              if P  > 0 else 0.0   # recall / sensitivity
    fpr  = fp / N              if N  > 0 else 0.0   # fall-out
    tnr  = tn / N              if N  > 0 else 0.0   # specificity
    fnr  = fn / P              if P  > 0 else 0.0   # miss rate
    ppv  = tp / (tp + fp)      if (tp + fp) > 0 else 0.0  # precision
    npv  = tn / (tn + fn)      if (tn + fn) > 0 else 0.0
    fdr  = fp / (tp + fp)      if (tp + fp) > 0 else 0.0
    acc  = (tp + tn) / (P + N) if (P + N)  > 0 else 0.0
    ba   = (tpr + tnr) / 2
    f1   = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) > 0 else 0.0
    mcc_ = mcc(tp, fp, fn, tn)

    tp_scores = [m["similarity_score"] for m in matches
                 if norm_id(m["corpus_id"]) in reference_ids]
    fp_scores = [m["similarity_score"] for m in matches
                 if norm_id(m["corpus_id"]) not in reference_ids]

    # Rank buckets for TP
    rank_buckets = {}
    for (lo, hi) in [(1, 10), (11, 25), (26, 50), (51, 75), (76, 101)]:
        rank_buckets[f"{lo}-{hi}"] = sum(
            1 for m in matches
            if lo <= m["rank"] <= hi and norm_id(m["corpus_id"]) in reference_ids
        )

    # ROC curve: sweep rank threshold from 1 to len(matches)
    roc_tpr, roc_fpr = [0.0], [0.0]
    tp_c, fp_c = 0, 0
    for m in matches:
        if norm_id(m["corpus_id"]) in reference_ids:
            tp_c += 1
        else:
            fp_c += 1
        roc_tpr.append(tp_c / P if P > 0 else 0.0)
        roc_fpr.append(fp_c / N if N > 0 else 0.0)

    # Partial AUC over the swept ranks (trapezoidal)
    auc = float(np.trapezoid(roc_tpr, roc_fpr))

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        P=P, N=N,
        tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr,
        ppv=ppv, npv=npv, fdr=fdr,
        acc=acc, ba=ba, f1=f1, mcc=mcc_,
        tp_scores=tp_scores, fp_scores=fp_scores,
        rank_buckets=rank_buckets,
        roc_tpr=roc_tpr, roc_fpr=roc_fpr,
        auc=auc,
        top_k=len(matches),
        n_reference=P,
        total_candidates=total,
    )


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_confusion_matrix(r: dict) -> None:
    tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
    P, N = r["P"], r["N"]
    print(f"\n  Confusion matrix — {r['label_short']}  "
          f"(P={P}, N={N}, total={r['total_candidates']})")
    print(f"  {'':25s} {'Pred +':>10} {'Pred -':>10} {'Total':>10}")
    print(f"  {'Actual + (reference)':25s} {tp:>10} {fn:>10} {P:>10}")
    print(f"  {'Actual - (non-ref)':25s} {fp:>10} {tn:>10} {N:>10}")
    print(f"  {'Total retrieved/not':25s} {tp+fp:>10} {fn+tn:>10} {P+N:>10}")


def print_metrics_table(results: list) -> None:
    cols = ["label_short", "tp", "fp", "fn", "tn",
            "tpr", "fpr", "tnr", "ppv", "f1", "ba", "mcc", "auc"]
    widths = [22, 4, 4, 4, 6, 7, 8, 7, 7, 7, 7, 7, 7]
    hdrs   = ["Pipeline", "TP", "FP", "FN", "TN",
              "TPR", "FPR", "TNR", "PPV", "F1", "BalAcc", "MCC", "pAUC"]

    header = "".join(f"{h:>{w}}" for h, w in zip(hdrs, widths))
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = [
            r["label_short"],
            str(r["tp"]), str(r["fp"]), str(r["fn"]), str(r["tn"]),
            f"{r['tpr']:.3f}", f"{r['fpr']:.5f}", f"{r['tnr']:.4f}",
            f"{r['ppv']:.3f}", f"{r['f1']:.3f}", f"{r['ba']:.4f}",
            f"{r['mcc']:.4f}", f"{r['auc']:.5f}",
        ]
        print("".join(f"{v:>{w}}" for v, w in zip(row, widths)))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reference_ids = load_reference(REFERENCE_FASTA)
    print(f"Reference sequences (seqhub, excl. query): {len(reference_ids)}")

    results = []
    for cfg in PIPELINES:
        if not Path(cfg["file"]).exists():
            print(f"  SKIP (missing): {cfg['file']}")
            continue
        ev = evaluate(cfg["file"], reference_ids)
        results.append({**cfg, **ev})

    if not results:
        print("No result files found — nothing to plot.")
        return

    # Print confusion matrices
    for r in results:
        print_confusion_matrix(r)

    # Print extended metrics table
    print_metrics_table(results)

    # Rank distribution for best pipeline
    best = max(results, key=lambda r: r["tp"])
    print(f"Rank distribution of TP hits — {best['label_short']}:")
    for bucket, count in best["rank_buckets"].items():
        print(f"  rank {bucket:<6} {count:>3}  {'█' * count}")
    print()

    # -----------------------------------------------------------------------
    # Figure 0 — Summary card (self-contained, judge-friendly)
    # -----------------------------------------------------------------------
    best_r = results[-1]  # last pipeline = best (full-seq+prefix)
    tp, fp, fn, tn = best_r["tp"], best_r["fp"], best_r["fn"], best_r["tn"]
    P, N = best_r["P"], best_r["N"]
    total = best_r["total_candidates"]

    fig0, ax0 = plt.subplots(figsize=(9, 6))
    ax0.axis("off")
    fig0.patch.set_facecolor("#f8f9fa")

    # Title block
    ax0.text(0.5, 0.97, "gLM2 + DIAMOND  —  Protein Sequence Search Benchmark",
             ha="center", va="top", fontsize=14, fontweight="bold", transform=ax0.transAxes)
    ax0.text(0.5, 0.91,
             "Query: P02981 (TetA class C tetracycline efflux transporter)  ·  "
             "Pipeline: DIAMOND pre-filter (ultra-sensitive, 50 k hits) → gLM2 cosine re-rank",
             ha="center", va="top", fontsize=9, color="#444444", transform=ax0.transAxes)
    ax0.text(0.5, 0.86,
             f"Candidate pool: {total:,} DIAMOND hits from OG_prot90 (85 M metagenome proteins)  ·  "
             f"Ground truth: {P} seqhub reference sequences  ·  Retrieval cutoff: top-{tp+fp}",
             ha="center", va="top", fontsize=9, color="#444444", transform=ax0.transAxes)

    # Confusion matrix table
    col_labels = ["", "Predicted +\n(in top-101)", "Predicted −\n(not retrieved)", "Row total"]
    row_data = [
        ["Actual +  (reference seq)", f"TP = {tp}", f"FN = {fn}", f"P = {P}"],
        ["Actual −  (non-reference)", f"FP = {fp}", f"TN = {tn:,}", f"N = {N:,}"],
        ["Column total",              f"{tp+fp}",    f"{fn+tn:,}",   f"{P+N:,}"],
    ]
    cell_colors = [
        ["#e8f5e9", "#2ca02c", "#ffcdd2", "#f0f0f0"],
        ["#ffcdd2", "#d62728", "#c8e6c9", "#f0f0f0"],
        ["#f0f0f0", "#f0f0f0", "#f0f0f0", "#e0e0e0"],
    ]
    # Override text colors for dark cells
    text_colors = [
        ["black", "white", "black", "black"],
        ["black", "white", "black", "black"],
        ["black", "black", "black", "black"],
    ]

    tbl = ax0.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.24, 1.0, 0.52],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    # Widen first column so row labels aren't clipped
    for row_idx in range(len(row_data) + 1):
        tbl[row_idx, 0].set_width(0.32)
        for col_idx in range(1, 4):
            tbl[row_idx, col_idx].set_width(0.227)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#888888")
        if row == 0:  # header row
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(cell_colors[row - 1][col])
            cell.set_text_props(color=text_colors[row - 1][col],
                                fontweight="bold" if col in (1, 2) else "normal")
        cell.set_linewidth(0.8)

    # Key metrics strip
    metrics = [
        ("Precision (PPV)", f"{best_r['ppv']:.1%}"),
        ("Recall (TPR)",     f"{best_r['tpr']:.1%}"),
        ("F1 score",         f"{best_r['f1']:.1%}"),
        ("MCC",              f"{best_r['mcc']:.3f}"),
        ("Balanced Acc.",    f"{best_r['ba']:.1%}"),
        ("FPR",              f"{best_r['fpr']:.5f}"),
    ]
    n_m = len(metrics)
    for i, (lbl, val) in enumerate(metrics):
        x = (i + 0.5) / n_m
        ax0.text(x, 0.18, val, ha="center", va="center", fontsize=13,
                 fontweight="bold", color="#1a5276", transform=ax0.transAxes)
        ax0.text(x, 0.11, lbl, ha="center", va="center", fontsize=8,
                 color="#555555", transform=ax0.transAxes)

    ax0.plot([0.01, 0.99], [0.21, 0.21], color="#cccccc", lw=1,
             transform=ax0.transAxes, clip_on=False)
    ax0.text(0.5, 0.04,
             f"Pipeline: {best_r['label_short']}   ·   "
             f"TP+FP+FN+TN = {tp}+{fp}+{fn}+{tn:,} = {P+N:,}",
             ha="center", va="bottom", fontsize=8, color="#888888",
             transform=ax0.transAxes)

    plt.tight_layout()
    out0 = OUT_DIR / "summary_card.png"
    plt.savefig(out0, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out0}")

    # -----------------------------------------------------------------------
    # Figure 1 — 2×2 Confusion matrices (one per pipeline)
    # -----------------------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion matrices  (P=100 reference seqs, N=49,900 non-reference candidates)",
                 fontsize=11, fontweight="bold")

    cmap_g = LinearSegmentedColormap.from_list("g", ["#ffffff", "#2ca02c"])
    cmap_r = LinearSegmentedColormap.from_list("r", ["#ffffff", "#d62728"])

    for ax, r in zip(axes, results):
        mat = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]])
        # Use two colormaps: green for diagonal (correct), red for off-diagonal
        colors = np.array([
            [cmap_g(r["tp"] / 100), cmap_r(r["fn"] / 100)],
            [cmap_r(r["fp"] / 100), cmap_g(r["tn"] / r["N"])],
        ])
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                           facecolor=colors[i, j], edgecolor="black", lw=1.5))
                lbl = ["TP", "FN", "FP", "TN"][i * 2 + j]
                val = mat[i, j]
                ax.text(j + 0.5, 1.5 - i, f"{lbl}\n{val:,}",
                        ha="center", va="center", fontsize=13, fontweight="bold")

        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Pred +", "Pred −"])
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Actual −", "Actual +"])
        ax.set_title(f"{r['label_short']}\n"
                     f"TPR={r['tpr']:.2f}  FPR={r['fpr']:.5f}  MCC={r['mcc']:.3f}",
                     fontsize=9)
        ax.tick_params(length=0)

    plt.tight_layout()
    out1 = OUT_DIR / "confusion_matrices.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # -----------------------------------------------------------------------
    # Figure 2 — ROC curves (rank sweep)
    # -----------------------------------------------------------------------
    fig2, ax_roc = plt.subplots(figsize=(7, 6))
    for r in results:
        ax_roc.plot(r["roc_fpr"], r["roc_tpr"],
                    color=r["color"], lw=2.0,
                    label=f"{r['label_short']}  (pAUC={r['auc']:.4f})")
        # Mark operating point (last rank = top_k)
        ax_roc.scatter(r["roc_fpr"][-1], r["roc_tpr"][-1],
                       color=r["color"], s=80, zorder=5, edgecolors="black", lw=0.5)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.5)")
    ax_roc.set_xlabel("FPR  (False Positive Rate = FP / N)")
    ax_roc.set_ylabel("TPR  (True Positive Rate = TP / P = Recall)")
    ax_roc.set_title("ROC curve — rank swept from 1 to top-k\n"
                     "(filled dot = operating point at top-101;\n"
                     " x-axis zoomed to FPR ≤ 0.002 since N=49,900)")
    ax_roc.legend(fontsize=9)
    max_fpr = max(r["roc_fpr"][-1] for r in results) * 1.4
    ax_roc.set_xlim(-max_fpr * 0.02, max_fpr)
    ax_roc.set_ylim(0, 1.05)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    out2 = OUT_DIR / "roc_curve.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # -----------------------------------------------------------------------
    # Figure 3 — TP/FP/FN bar comparison + Precision/Recall/F1/MCC
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("gLM2 + DIAMOND pipeline — Retrieval benchmark\n"
                  "(query: P02981 TetA class C  |  reference: seqhub top-100)",
                  fontsize=11, fontweight="bold")

    ax = axes3[0]
    labels = [r["label"] for r in results]
    tp_vals = [r["tp"] for r in results]
    fp_vals = [r["fp"] for r in results]
    fn_vals = [r["fn"] for r in results]
    x = np.arange(len(results))
    w = 0.25
    bars_tp = ax.bar(x - w, tp_vals, w, label="TP (found)", color="#2ca02c", edgecolor="white")
    bars_fp = ax.bar(x,     fp_vals, w, label="FP (wrong)", color="#d62728", edgecolor="white")
    bars_fn = ax.bar(x + w, fn_vals, w, label="FN (missed)", color="#aec7e8", edgecolor="white")
    for bar in list(bars_tp) + list(bars_fp) + list(bars_fn):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, str(int(h)),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count  (top-101 candidates)")
    ax.set_title("TP / FP / FN  across pipeline variants")
    ax.axhline(len(reference_ids), color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(tp_vals) + 12)

    ax2 = axes3[1]
    metric_labels = ["Precision\n(PPV)", "Recall\n(TPR)", "F1", "MCC", "Bal.Acc"]
    n_pipelines = len(results)
    x2 = np.arange(len(metric_labels))
    bar_w = 0.8 / n_pipelines
    for i, r in enumerate(results):
        vals = [r["ppv"], r["tpr"], r["f1"], r["mcc"], r["ba"]]
        offset = (i - n_pipelines / 2 + 0.5) * bar_w
        bars = ax2.bar(x2 + offset, vals, bar_w, label=r["label_short"],
                       color=r["color"], edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x2); ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.08)
    ax2.set_title("Classification metrics  @  top-101")
    ax2.legend(fontsize=9)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    out3 = OUT_DIR / "benchmark_comparison.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # -----------------------------------------------------------------------
    # Figure 4 — Score distribution for best pipeline (TP vs FP)
    # -----------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    bins = np.linspace(
        min(min(best["tp_scores"]), min(best["fp_scores"])) - 0.0005,
        max(max(best["tp_scores"]), max(best["fp_scores"])) + 0.0005,
        40,
    )
    ax4.hist(best["tp_scores"], bins=bins, alpha=0.7, color="#2ca02c",
             label=f"TP  (n={len(best['tp_scores'])})")
    ax4.hist(best["fp_scores"], bins=bins, alpha=0.7, color="#d62728",
             label=f"FP  (n={len(best['fp_scores'])})")
    ax4.set_xlabel("Cosine similarity score")
    ax4.set_ylabel("Count")
    ax4.set_title(f"Score distribution — {best['label_short']}\n"
                  f"green = seqhub reference hits (TP),  red = wrong hits in top-101 (FP)")
    ax4.legend()
    plt.tight_layout()
    out4 = OUT_DIR / "score_distribution.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")

    # -----------------------------------------------------------------------
    # Figure 5 — Rank heatmap across all pipelines
    # -----------------------------------------------------------------------
    bucket_keys = list(results[0]["rank_buckets"].keys())
    fig5, ax5 = plt.subplots(figsize=(8, 3.5))
    data_matrix = np.array([[r["rank_buckets"][k] for k in bucket_keys] for r in results])
    im = ax5.imshow(data_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=max(data_matrix.max(), 1))
    ax5.set_xticks(range(len(bucket_keys)))
    ax5.set_xticklabels([f"rank\n{k}" for k in bucket_keys])
    ax5.set_yticks(range(len(results)))
    ax5.set_yticklabels([r["label_short"] for r in results])
    ax5.set_title("TP count by rank bucket  (darker = more hits found)")
    for i in range(len(results)):
        for j in range(len(bucket_keys)):
            val = int(data_matrix[i, j])
            ax5.text(j, i, str(val), ha="center", va="center", fontsize=11,
                     fontweight="bold",
                     color="white" if val > data_matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax5, shrink=0.8, label="TP count")
    plt.tight_layout()
    out5 = OUT_DIR / "rank_heatmap.png"
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out5}")


if __name__ == "__main__":
    main()
