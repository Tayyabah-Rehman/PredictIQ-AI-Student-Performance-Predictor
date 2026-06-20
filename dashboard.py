"""
dashboard.py — Analytics Dashboard Generator
Generates all charts using Matplotlib + Seaborn and saves to static/img/
Run once after train_model.py to generate dashboard assets.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "static", "img")
os.makedirs(OUT, exist_ok=True)

# ── Palette (matches dark UI) ──────────────────────────────────
BG       = "#161b27"
BG2      = "#1c2333"
BORDER   = "#2a3347"
TEXT     = "#e8ecf4"
MUTED    = "#8896b0"
ACCENT   = "#38bdf8"
COLORS   = {"Excellent":"#34d399","Good":"#60a5fa","Average":"#fbbf24","Needs Improvement":"#f87171"}

def dark_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.6, alpha=0.7)
    return fig, ax

def dark_fig_multi(rows, cols, w=14, h=10):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(BG2)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, axes

# ── Generate dataset (same logic as train_model) ──────────────
def make_data(n=8000):
    rng = np.random.default_rng(42)
    study_hours     = rng.beta(2.5, 2, n) * 12
    attendance      = np.clip(rng.beta(4,1.5,n)*100 + study_hours*1.5 + rng.normal(0,2,n), 0, 100)
    assignments     = rng.integers(0, 11, n).astype(float)
    prev_marks      = np.clip(rng.normal(62,14,n) + study_hours*2.5 + assignments*1.8 + rng.normal(0,4,n), 0, 100)
    sleep_hours     = np.clip(rng.normal(7,1.2,n), 4, 10)
    extracurricular = rng.integers(0, 6, n).astype(float)
    score = (prev_marks*0.38 + attendance*0.22 + study_hours*3.8
             + assignments*2.2 + sleep_hours*1.5 + extracurricular*0.8
             + rng.normal(0,3,n))
    def grade(s):
        if s>=85: return "Excellent"
        elif s>=68: return "Good"
        elif s>=52: return "Average"
        else: return "Needs Improvement"
    return pd.DataFrame({
        "study_hours": study_hours.round(2), "attendance": attendance.round(2),
        "assignments": assignments.astype(int), "previous_marks": prev_marks.round(2),
        "sleep_hours": sleep_hours.round(2), "extracurricular": extracurricular.astype(int),
        "grade": [grade(s) for s in score]
    })

# ══════════════════════════════════════════════════════════════
# CHART 1 — Attendance Distribution (histogram + KDE)
# ══════════════════════════════════════════════════════════════
def chart_attendance(df):
    fig, ax = dark_fig(10, 5)
    grade_order = ["Excellent","Good","Average","Needs Improvement"]
    for g in grade_order:
        sub = df[df["grade"]==g]["attendance"]
        ax.hist(sub, bins=30, alpha=0.55, color=COLORS[g], edgecolor="none", label=g)
        sub.plot.kde(ax=ax, color=COLORS[g], linewidth=2)
    ax.set_xlabel("Attendance (%)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Attendance Distribution by Performance Category", fontsize=13, fontweight="bold", pad=14)
    legend = ax.legend(fontsize=9, framealpha=0.15, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_attendance.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ chart_attendance.png")

# ══════════════════════════════════════════════════════════════
# CHART 2 — Marks Distribution (violin + box)
# ══════════════════════════════════════════════════════════════
def chart_marks(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    grade_order = ["Excellent","Good","Average","Needs Improvement"]

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG2)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)

    palette = {g: COLORS[g] for g in grade_order}

    # Violin
    sns.violinplot(data=df, x="grade", y="previous_marks", order=grade_order,
                   palette=palette, ax=ax1, inner="quartile", linewidth=1.2)
    ax1.set_title("Previous Marks — Violin Plot", fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax1.set_xlabel("Performance Category", fontsize=10, color=MUTED)
    ax1.set_ylabel("Previous Marks (/100)", fontsize=10, color=MUTED)
    ax1.set_xticklabels(ax1.get_xticklabels(), color=MUTED, fontsize=9)
    ax1.set_yticklabels(ax1.get_yticklabels(), color=MUTED)

    # Box
    bp = ax2.boxplot([df[df["grade"]==g]["previous_marks"].values for g in grade_order],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color=TEXT, linewidth=2),
                     whiskerprops=dict(color=MUTED, linewidth=1.2),
                     capprops=dict(color=MUTED, linewidth=1.2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.3))
    for patch, g in zip(bp["boxes"], grade_order):
        patch.set_facecolor(COLORS[g]+"55"); patch.set_edgecolor(COLORS[g])
    ax2.set_xticks(range(1, len(grade_order)+1))
    ax2.set_xticklabels(grade_order, color=MUTED, fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), color=MUTED)
    ax2.set_title("Previous Marks — Box Plot", fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax2.set_xlabel("Performance Category", fontsize=10, color=MUTED)
    ax2.set_ylabel("Previous Marks (/100)", fontsize=10, color=MUTED)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUT, "chart_marks.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ chart_marks.png")

# ══════════════════════════════════════════════════════════════
# CHART 3 — Feature Correlation Heatmap
# ══════════════════════════════════════════════════════════════
def chart_heatmap(df):
    numeric_cols = ["study_hours","attendance","assignments","previous_marks",
                    "sleep_hours","extracurricular"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True   # upper triangle only

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.8, linecolor=BG,
                annot_kws={"size":10,"color":TEXT},
                cbar_kws={"shrink":0.8}, ax=ax)

    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold",
                 color=TEXT, pad=14)
    ax.tick_params(colors=MUTED, labelsize=10)
    labels = ["Study Hrs","Attendance","Assignments","Prev Marks","Sleep Hrs","Extracurric"]
    ax.set_xticklabels(labels, color=MUTED, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(labels, color=MUTED, fontsize=9, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_heatmap.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ chart_heatmap.png")

# ══════════════════════════════════════════════════════════════
# CHART 4 — Performance Categories (donut + bar)
# ══════════════════════════════════════════════════════════════
def chart_categories(df):
    grade_order = ["Excellent","Good","Average","Needs Improvement"]
    counts = df["grade"].value_counts().reindex(grade_order)
    pcts   = counts / counts.sum() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # Donut
    ax1.set_facecolor(BG)
    wedge_colors = [COLORS[g] for g in grade_order]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
    ax1.legend(wedges, grade_order, loc="lower center", fontsize=9,
               framealpha=0.15, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT,
               bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax1.set_title("Grade Distribution (Donut)", fontsize=12, fontweight="bold",
                  color=TEXT, pad=10)

    # Horizontal bar
    ax2.set_facecolor(BG2)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
    bars = ax2.barh(grade_order, counts.values,
                    color=[COLORS[g]+"bb" for g in grade_order],
                    edgecolor=[COLORS[g] for g in grade_order], linewidth=1.5, height=0.55)
    for bar, val, pct in zip(bars, counts.values, pcts.values):
        ax2.text(bar.get_width()+30, bar.get_y()+bar.get_height()/2,
                 f"{val:,}  ({pct:.1f}%)", va="center", color=TEXT, fontsize=10)
    ax2.set_xlabel("Number of Students", fontsize=10, color=MUTED)
    ax2.set_title("Student Count per Category", fontsize=12, fontweight="bold", color=TEXT, pad=10)
    ax2.tick_params(colors=MUTED, labelsize=10)
    ax2.set_yticklabels(grade_order, color=MUTED)
    ax2.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6, axis="x")
    ax2.set_xlim(0, counts.max()*1.28)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUT, "chart_categories.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ chart_categories.png")

# ══════════════════════════════════════════════════════════════
# CHART 5 — Study Hours vs Marks Scatter (bonus)
# ══════════════════════════════════════════════════════════════
def chart_scatter(df):
    fig, ax = dark_fig(10, 5)
    grade_order = ["Needs Improvement","Average","Good","Excellent"]
    for g in grade_order:
        sub = df[df["grade"]==g]
        ax.scatter(sub["study_hours"], sub["previous_marks"],
                   c=COLORS[g], alpha=0.35, s=18, label=g, edgecolors="none")
    ax.set_xlabel("Daily Study Hours", fontsize=11)
    ax.set_ylabel("Previous Marks (/100)", fontsize=11)
    ax.set_title("Study Hours vs Previous Marks (coloured by Grade)", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9, framealpha=0.15, facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_scatter.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  ✓ chart_scatter.png")

# ══════════════════════════════════════════════════════════════
# SUMMARY STATS JSON (for dashboard API)
# ══════════════════════════════════════════════════════════════
def save_stats(df):
    grade_order = ["Excellent","Good","Average","Needs Improvement"]
    stats = {
        "total_records":    int(len(df)),
        "avg_marks":        round(float(df["previous_marks"].mean()), 2),
        "avg_attendance":   round(float(df["attendance"].mean()), 2),
        "avg_study_hours":  round(float(df["study_hours"].mean()), 2),
        "avg_sleep":        round(float(df["sleep_hours"].mean()), 2),
        "grade_counts":     df["grade"].value_counts().reindex(grade_order).to_dict(),
        "grade_pcts":       (df["grade"].value_counts(normalize=True).reindex(grade_order)*100).round(1).to_dict(),
        "marks_by_grade":   {g: round(float(df[df["grade"]==g]["previous_marks"].mean()),1) for g in grade_order},
        "attend_by_grade":  {g: round(float(df[df["grade"]==g]["attendance"].mean()),1) for g in grade_order},
        "study_by_grade":   {g: round(float(df[df["grade"]==g]["study_hours"].mean()),1) for g in grade_order},
        "correlation":      df[["study_hours","attendance","assignments","previous_marks",
                                 "sleep_hours","extracurricular"]].corr().round(3).to_dict(),
    }
    path = os.path.join(os.path.dirname(__file__), "model", "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  ✓ dataset_stats.json")
    return stats

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*55)
    print("  GENERATING ANALYTICS DASHBOARD ASSETS")
    print("="*55)
    df = make_data(8000)
    stats = save_stats(df)
    print(f"\nDataset: {stats['total_records']:,} records")
    print(f"  Avg Marks      : {stats['avg_marks']}")
    print(f"  Avg Attendance : {stats['avg_attendance']}%")
    print(f"  Avg Study Hrs  : {stats['avg_study_hours']}")
    print("\nGenerating charts...")
    chart_attendance(df)
    chart_marks(df)
    chart_heatmap(df)
    chart_categories(df)
    chart_scatter(df)
    print("\n✓ All assets saved to static/img/")
    print("="*55)
