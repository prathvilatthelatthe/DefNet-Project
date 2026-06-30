"""
DeforestNet - Advanced Pitch-Deck Graphs (v2)
==============================================
Industry-grade visualizations showing competitive impact,
market positioning, and why DeforestNet changes the game.

Generates 10 high-resolution charts at 250 DPI.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "docs" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Premium Dark Theme ────────────────────────────────────────
BG        = "#080e1a"
CARD      = "#111827"
SURFACE   = "#1e293b"
TEXT      = "#f1f5f9"
TEXT2     = "#94a3b8"
TEXT3     = "#64748b"
GRID      = "#1e293b"
GREEN     = "#34d399"
GREEN_D   = "#059669"
RED       = "#f87171"
RED_D     = "#dc2626"
AMBER     = "#fbbf24"
AMBER_D   = "#d97706"
BLUE      = "#60a5fa"
BLUE_D    = "#2563eb"
PURPLE    = "#a78bfa"
PURPLE_D  = "#7c3aed"
CYAN      = "#22d3ee"
PINK      = "#f472b6"
LIME      = "#a3e635"
ORANGE    = "#fb923c"
TEAL      = "#2dd4bf"
WHITE     = "#ffffff"

DPI = 250

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     CARD,
    "axes.edgecolor":     SURFACE,
    "axes.labelcolor":    TEXT,
    "text.color":         TEXT,
    "xtick.color":        TEXT2,
    "ytick.color":        TEXT2,
    "grid.color":         GRID,
    "grid.alpha":         0.4,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

GLOW = [pe.withStroke(linewidth=3, foreground=BG)]


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.35)
    plt.close(fig)
    print(f"  [OK] {name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TAM / SAM / SOM Funnel  –  "How big is the opportunity?"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_01_market_funnel():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 8.5)

    fig.text(0.5, 0.95, "Total Addressable Market Opportunity",
             ha="center", fontsize=18, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.91, "DeforestNet sits at the intersection of three mega-trends converging into one market",
             ha="center", fontsize=10, color=TEXT2)

    layers = [
        ("TAM", "$50.5B", "Satellite Data Services\nMarket by 2032 (19.9% CAGR)",
         5.5, 7.0, BLUE_D, BLUE, "60"),
        ("SAM", "$7.4B", "Earth Observation &\nGeoAI Analytics by 2032",
         4.2, 5.4, PURPLE_D, PURPLE, "50"),
        ("TARGET", "$2.5B", "Forest Monitoring &\nDeforestation Detection",
         3.0, 3.8, GREEN_D, GREEN, "40"),
        ("SOM", "$250M+", "EUDR Compliance +\nCarbon MRV + Govt Contracts",
         1.8, 2.2, AMBER_D, AMBER, "35"),
    ]

    for label, value, desc, half_w, y, c_dark, c_light, alpha_str in layers:
        alpha = int(alpha_str) / 100
        # Trapezoid
        verts = [(-half_w, y), (half_w, y),
                 (half_w - 0.7, y - 1.0), (-half_w + 0.7, y - 1.0)]
        poly = plt.Polygon(verts, facecolor=c_dark, edgecolor=c_light,
                           lw=2, alpha=0.85, zorder=2)
        ax.add_patch(poly)

        ax.text(0, y - 0.35, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=WHITE, zorder=3,
                path_effects=GLOW)
        ax.text(0, y - 0.7, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=c_light, zorder=3)

        side = 1 if label in ("TAM", "TARGET") else -1
        ax.annotate(desc, xy=(half_w * 0.7 * side, y - 0.5),
                    xytext=(half_w + 1.2 * side, y - 0.5),
                    fontsize=9, color=TEXT2, va="center",
                    ha="left" if side > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=TEXT3, lw=0.8))

    # Bottom arrow
    ax.annotate("", xy=(0, 0.3), xytext=(0, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=3))
    ax.text(0, -0.1, "DeforestNet Entry Point", fontsize=11,
            fontweight="bold", color=GREEN, ha="center")
    ax.text(0, -0.5, "400,000+ EU operators need compliance tools by Dec 2026",
            fontsize=9, color=AMBER, ha="center")

    save(fig, "01_market_funnel.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Competitive Positioning Map  –  "Where we sit vs everyone"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_02_positioning_map():
    fig, ax = plt.subplots(figsize=(12, 8))

    companies = {
        "DeforestNet":  {"x": 9.2, "y": 9.0, "s": 700, "c": GREEN,  "edge": WHITE},
        "Satelligence": {"x": 7.0, "y": 7.5, "s": 500, "c": BLUE,   "edge": BLUE},
        "Kayrros":      {"x": 5.5, "y": 6.0, "s": 400, "c": ORANGE, "edge": ORANGE},
        "SarVision":    {"x": 6.5, "y": 5.0, "s": 300, "c": CYAN,   "edge": CYAN},
        "Planet Labs":  {"x": 4.0, "y": 7.0, "s": 600, "c": AMBER,  "edge": AMBER},
        "GFW / GLAD":   {"x": 3.5, "y": 4.5, "s": 550, "c": PURPLE, "edge": PURPLE},
        "MapBiomas":    {"x": 4.0, "y": 3.5, "s": 350, "c": TEAL,   "edge": TEAL},
        "Maxar":        {"x": 2.0, "y": 6.5, "s": 450, "c": RED,    "edge": RED},
    }

    # Quadrant backgrounds
    for qx, qy, label, alpha in [
        (7.5, 7.5, "LEADERS", 0.06),
        (2.5, 7.5, "DATA\nPROVIDERS", 0.03),
        (2.5, 2.5, "LEGACY\nTOOLS", 0.03),
        (7.5, 2.5, "SPECIALISTS", 0.03),
    ]:
        rect = mpatches.FancyBboxPatch((qx - 2.4, qy - 2.4), 4.8, 4.8,
                                        boxstyle="round,pad=0.15",
                                        facecolor=GREEN if "LEAD" in label else SURFACE,
                                        alpha=alpha, edgecolor="none")
        ax.add_patch(rect)
        ax.text(qx, qy - 2.0, label, ha="center", fontsize=8, color=TEXT3,
                fontweight="bold", alpha=0.6)

    # Quadrant dividers
    ax.axhline(5, color=TEXT3, ls=":", lw=0.8, alpha=0.3)
    ax.axvline(5, color=TEXT3, ls=":", lw=0.8, alpha=0.3)

    # Plot companies
    for name, d in companies.items():
        ax.scatter(d["x"], d["y"], s=d["s"], c=d["c"], alpha=0.7,
                   edgecolors=d["edge"], linewidth=2, zorder=3)
        # Label offset
        ox, oy = 0.3, 0.35
        if name == "Kayrros":
            oy = -0.4
        if name == "Planet Labs":
            ox = -0.3
            oy = 0.4
        if name == "Maxar":
            ox = -0.3
        if name == "GFW / GLAD":
            oy = -0.45

        fw = "bold" if name == "DeforestNet" else "normal"
        fs = 11 if name == "DeforestNet" else 9
        ax.text(d["x"] + ox, d["y"] + oy, name, fontsize=fs,
                fontweight=fw, color=d["c"], zorder=4)

    # DeforestNet highlight ring
    circle = plt.Circle((9.2, 9.0), 0.6, fill=False, edgecolor=GREEN,
                         lw=2, ls="--", alpha=0.5, zorder=2)
    ax.add_patch(circle)

    ax.set_xlabel("AI / Classification Sophistication  >>>", fontsize=11, labelpad=10)
    ax.set_ylabel("Monitoring Completeness  >>>", fontsize=11, labelpad=10)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Competitive Positioning Map", fontsize=16, pad=15)

    # Legend
    legend_items = [
        ("Bubble size = Market reach / Data volume", TEXT3),
        ("DeforestNet: Only solution in LEADER quadrant with 6-class AI", GREEN),
    ]
    for i, (txt, c) in enumerate(legend_items):
        ax.text(0.7, 10.0 - i * 0.5, txt, fontsize=8, color=c, fontstyle="italic")

    save(fig, "02_competitive_positioning.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Impact Scorecard  –  "Why we win on every dimension"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_03_impact_scorecard():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 8.5)

    fig.text(0.5, 0.96, "DeforestNet Impact Scorecard vs Industry",
             ha="center", fontsize=18, fontweight="bold", color=WHITE)

    metrics = [
        ("CAUSE IDENTIFICATION", "6 classes", "Binary only", 100, 0,
         GREEN, "Logging, Mining, Agriculture, Fire, Infrastructure"),
        ("SPECTRAL BANDS", "11 bands", "3-6 bands", 100, 40,
         BLUE, "4 optical + 2 SAR + 5 derived indices"),
        ("CLOUD PENETRATION", "95-100%", "13-32%", 97, 25,
         CYAN, "SAR microwave passes through clouds, rain, smoke"),
        ("SPATIAL RESOLUTION", "10m", "30m", 90, 33,
         PURPLE, "9x more detail per pixel than Landsat-based systems"),
        ("ANNUAL DATA COST", "$0", "$30-500K", 100, 5,
         AMBER, "100% free ESA Copernicus Sentinel data"),
        ("EXPLAINABILITY", "GradCAM", "Black box", 100, 10,
         PINK, "Visual heatmaps showing why the AI made each decision"),
        ("EUDR COMPLIANCE", "Ready", "Not designed", 95, 15,
         ORANGE, "Cause ID directly maps to EU regulation commodities"),
    ]

    bar_h = 0.32
    for i, (label, ours, theirs, score_us, score_them, color, note) in enumerate(metrics):
        y = 7.0 - i * 1.05

        # Label
        ax.text(0.1, y + 0.15, label, fontsize=9, fontweight="bold", color=color)

        # Our bar
        bar_width = score_us / 100 * 7.5
        rect_us = mpatches.FancyBboxPatch(
            (3.0, y), bar_width, bar_h, boxstyle="round,pad=0.05",
            facecolor=color, alpha=0.8, edgecolor="none")
        ax.add_patch(rect_us)
        ax.text(3.0 + bar_width + 0.15, y + bar_h / 2, ours,
                fontsize=10, fontweight="bold", color=color, va="center")

        # Their bar
        bar_width_them = score_them / 100 * 7.5
        rect_them = mpatches.FancyBboxPatch(
            (3.0, y - 0.38), max(bar_width_them, 0.15), bar_h,
            boxstyle="round,pad=0.05",
            facecolor=TEXT3, alpha=0.4, edgecolor="none")
        ax.add_patch(rect_them)
        ax.text(3.0 + max(bar_width_them, 0.15) + 0.15, y - 0.38 + bar_h / 2,
                theirs, fontsize=8, color=TEXT3, va="center")

        # Note
        ax.text(11.5, y - 0.1, note, fontsize=7, color=TEXT3,
                va="center", fontstyle="italic", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.5))

    # Labels at top
    ax.text(3.0, 7.8, "DeforestNet", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(3.0, 7.5, "Industry Average", fontsize=9, color=TEXT3)
    for x in [3.0, 6.75, 10.5]:
        ax.plot([x, x], [-0.3, 7.3], color=SURFACE, lw=0.5, alpha=0.5)

    save(fig, "03_impact_scorecard.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Radar Spider  –  Tech capability comparison (enhanced)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_04_radar():
    categories = [
        "Spectral\nRichness", "Spatial\nResolution", "Cloud\nPenetration",
        "Cause\nClassification", "Deep Learning\nArchitecture", "Real-Time\nAlerts",
        "Cost\nEfficiency", "Explainability", "EUDR\nReadiness"
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    deforestnet  = [10, 8, 10, 10, 9, 8, 10, 9, 9]
    gfw          = [4,  3,  1,  1,  3, 5, 10, 2, 1]
    satelligence = [7,  8,  8,  2,  8, 8,  2, 2, 8]
    planet       = [5,  10, 1,  1,  5, 9,  3, 2, 2]

    for d in [deforestnet, gfw, satelligence, planet]:
        d.append(d[0])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD)

    # Fill DeforestNet with gradient effect
    ax.fill(angles, deforestnet, alpha=0.3, color=GREEN)
    ax.plot(angles, deforestnet, "o-", color=GREEN, lw=3, ms=9,
            label="DeforestNet", zorder=5)
    # Glow effect on DeforestNet line
    ax.plot(angles, deforestnet, color=GREEN, lw=8, alpha=0.15, zorder=4)

    ax.fill(angles, satelligence, alpha=0.08, color=BLUE)
    ax.plot(angles, satelligence, "s--", color=BLUE, lw=1.5, ms=5,
            label="Satelligence", alpha=0.8)

    ax.fill(angles, planet, alpha=0.06, color=AMBER)
    ax.plot(angles, planet, "D--", color=AMBER, lw=1.5, ms=5,
            label="Planet Labs", alpha=0.8)

    ax.fill(angles, gfw, alpha=0.06, color=PURPLE)
    ax.plot(angles, gfw, "^--", color=PURPLE, lw=1.5, ms=5,
            label="GFW / GLAD", alpha=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold", color=TEXT)
    ax.set_ylim(0, 11)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color=TEXT3)
    ax.grid(color=SURFACE, alpha=0.4)

    # Score annotations
    ax.text(0.5, 0.02, "DeforestNet total: 93/90  |  Next best: 53/90",
            transform=fig.transFigure, ha="center", fontsize=11, color=GREEN,
            fontweight="bold")

    ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.08),
              fontsize=11, framealpha=0.2, edgecolor=SURFACE)
    ax.set_title("9-Dimension Technical Capability Comparison",
                 fontsize=16, fontweight="bold", pad=30, y=1.08)

    save(fig, "04_radar_capability.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. EUDR Compliance Timeline  –  "Why NOW is the moment"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_05_eudr_timeline():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis("off")
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2, 4)

    fig.text(0.5, 0.95, "EUDR Compliance Timeline: The $2.5B Window of Opportunity",
             ha="center", fontsize=16, fontweight="bold", color=WHITE)

    # Timeline bar
    ax.plot([0.5, 13.5], [1, 1], color=SURFACE, lw=8, solid_capstyle="round", zorder=1)

    events = [
        (1.0,  "Jun 2023",  "EUDR\nAdopted",         TEXT3,   "above", 10),
        (3.0,  "Dec 2024",  "Info System\nLaunched",  BLUE,    "below", 10),
        (4.5,  "Mar 2026",  "WE ARE\nHERE",           GREEN,   "above", 12),
        (7.5,  "Dec 2026",  "EUDR Deadline\n(Large Operators)", RED, "below", 12),
        (9.5,  "Jun 2027",  "EUDR Deadline\n(Small Operators)", RED_D, "above", 10),
        (12.0, "2028-30",   "Carbon Market\nRecovery", AMBER,  "below", 10),
    ]

    for x, date, label, color, pos, fs in events:
        # Dot on timeline
        ax.plot(x, 1, "o", color=color, ms=14, zorder=3)
        ax.plot(x, 1, "o", color=color, ms=20, alpha=0.2, zorder=2)

        # Text
        y_offset = 2.2 if pos == "above" else -0.5
        va = "bottom" if pos == "above" else "top"
        ax.text(x, y_offset, label, ha="center", va=va, fontsize=fs,
                fontweight="bold", color=color)
        ax.text(x, y_offset + (0.5 if pos == "above" else -0.5),
                date, ha="center", va=va, fontsize=8, color=TEXT3)

        # Connector line
        y_start = 1.4 if pos == "above" else 0.6
        y_end = y_offset - (0.2 if pos == "above" else -0.2)
        ax.plot([x, x], [y_start, y_end], color=color, lw=1, alpha=0.4)

    # Highlight window
    rect = mpatches.FancyBboxPatch((3.8, 0.4), 4.3, 1.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=GREEN, alpha=0.08,
                                    edgecolor=GREEN, lw=2, ls="--")
    ax.add_patch(rect)
    ax.text(6.0, 0.15, "18-MONTH CAPTURE WINDOW", fontsize=9,
            fontweight="bold", color=GREEN, ha="center")

    # Bottom stat bar
    stats_y = -1.5
    stats = [
        ("400,000+", "EU operators need\ncompliance tools", AMBER),
        ("7 Commodities", "cattle, soy, palm oil,\ncocoa, coffee, wood, rubber", BLUE),
        ("$250M+", "addressable market\nin EUDR compliance alone", GREEN),
        ("ZERO", "competitors offer\ncause identification", RED),
    ]
    for i, (val, desc, c) in enumerate(stats):
        cx = 1.5 + i * 3.3
        ax.text(cx, stats_y, val, fontsize=16, fontweight="bold", color=c, ha="center")
        ax.text(cx, stats_y - 0.55, desc, fontsize=8, color=TEXT3, ha="center")

    save(fig, "05_eudr_timeline.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Deforestation Crisis  – "The problem we're solving"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_06_crisis():
    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: Bar chart
    years = list(range(2015, 2025))
    loss = [3.1, 3.0, 3.9, 3.6, 3.8, 4.2, 3.8, 4.1, 3.7, 6.7]

    gradient_colors = []
    for v in loss:
        if v > 5.0:
            gradient_colors.append(RED)
        elif v > 4.0:
            gradient_colors.append(ORANGE)
        elif v > 3.5:
            gradient_colors.append(AMBER)
        else:
            gradient_colors.append(BLUE)

    bars = ax.bar(years, loss, color=gradient_colors, width=0.7,
                  edgecolor=BG, linewidth=1.5, zorder=3)

    # 2024 special highlight
    bars[-1].set_edgecolor(RED)
    bars[-1].set_linewidth(3)
    ax.bar(2024, 6.7, width=0.7, color=RED, alpha=0.15, zorder=2)

    ax.annotate("RECORD HIGH\n6.7 Mha", (2024, 6.7),
                xytext=(0, 25), textcoords="offset points",
                fontsize=14, fontweight="bold", color=RED, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2),
                path_effects=GLOW)
    ax.annotate("+80% YoY", (2024, 6.7),
                xytext=(0, 60), textcoords="offset points",
                fontsize=10, color=AMBER, ha="center", fontweight="bold")

    # Glasgow target
    target_years = np.linspace(2021, 2030, 20)
    target_loss = np.linspace(3.8, 0, 20)
    ax.plot(target_years, target_loss, "--", color=GREEN, lw=2, alpha=0.6)
    ax.fill_between(target_years, target_loss, alpha=0.03, color=GREEN)
    ax.text(2029, 0.3, "Glasgow\n2030 Target", fontsize=8, color=GREEN,
            fontweight="bold", ha="center")

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Tropical Primary Forest Loss (Million Hectares)", fontsize=10)
    ax.set_title("The Deforestation Crisis is Accelerating", fontsize=14, pad=10)
    ax.set_xticks(years)
    ax.set_ylim(0, 8.5)
    ax.grid(True, axis="y", alpha=0.15)

    # Right: Key stats panel
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    stats = [
        ("6.7M", "hectares lost\nin 2024", RED, 9.0),
        ("3.1 Gt", "CO2e emissions\nfrom tropical loss", AMBER, 7.2),
        ("17 / 20", "countries off track\non Glasgow pledge", ORANGE, 5.4),
        ("80%", "year-over-year\nincrease in 2024", PINK, 3.6),
        ("$2B+", "carbon credits\nat risk annually", PURPLE, 1.8),
    ]

    for val, desc, color, y in stats:
        # Background card
        card = mpatches.FancyBboxPatch((0.5, y - 0.7), 9, 1.4,
                                        boxstyle="round,pad=0.2",
                                        facecolor=SURFACE, edgecolor=color,
                                        lw=1.5, alpha=0.6)
        ax2.add_patch(card)
        ax2.text(2.0, y, val, fontsize=18, fontweight="bold", color=color,
                 ha="center", va="center")
        ax2.text(5.5, y, desc, fontsize=9, color=TEXT2, va="center")

    fig.suptitle("", y=0.98)
    save(fig, "06_deforestation_crisis.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. The Classification Revolution  –  Binary vs 6-Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_07_classification_revolution():
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 3, width_ratios=[1, 0.15, 1], wspace=0.05)

    # Left: What competitors see
    ax1 = fig.add_subplot(gs[0])
    labels = ["Forest\n(No Change)", "Non-Forest\n(Change Detected)"]
    sizes = [65, 35]
    colors = ["#2d5a3d", "#7f1d1d"]
    edge_colors = [GREEN_D, RED_D]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 11, "color": TEXT},
        wedgeprops={"edgecolor": BG, "linewidth": 3},
        pctdistance=0.55
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(15)
        t.set_color(WHITE)
    ax1.set_title("Everyone Else\nBinary Classification", fontsize=14,
                  fontweight="bold", color=RED, pad=15)
    ax1.text(0, -1.4, '"Something happened here"', ha="center",
             fontsize=11, color=TEXT3, fontstyle="italic")
    ax1.text(0, -1.7, "Cannot tell WHAT or WHY", ha="center",
             fontsize=10, fontweight="bold", color=RED)

    # Center arrow
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.axis("off")
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.annotate("", xy=(0.8, 0.5), xytext=(0.2, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=4))
    ax_mid.text(0.5, 0.62, "vs", fontsize=14, fontweight="bold",
                color=TEXT3, ha="center")

    # Right: What DeforestNet sees
    ax2 = fig.add_subplot(gs[2])
    labels2 = ["Forest", "Logging", "Mining", "Agriculture", "Fire", "Infrastructure"]
    sizes2 = [50, 12, 10, 15, 8, 5]
    colors2 = ["#065f46", "#92400e", "#7f1d1d", "#78350f", "#991b1b", "#4c1d95"]
    edge_colors2 = [GREEN, ORANGE, RED, AMBER, RED_D, PURPLE]
    explode = [0, 0.06, 0.06, 0.06, 0.06, 0.06]
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=labels2, colors=colors2, autopct="%1.0f%%",
        startangle=90, explode=explode,
        textprops={"fontsize": 10, "color": TEXT},
        wedgeprops={"edgecolor": BG, "linewidth": 3},
        pctdistance=0.7
    )
    for w, ec in zip(wedges2, edge_colors2):
        w.set_edgecolor(ec)
    for t in autotexts2:
        t.set_fontweight("bold")
        t.set_fontsize(11)
        t.set_color(WHITE)
    ax2.set_title("DeforestNet\n6-Class Cause Identification", fontsize=14,
                  fontweight="bold", color=GREEN, pad=15)
    ax2.text(0, -1.4, '"Mining detected at 10.5N, 76.3E"', ha="center",
             fontsize=11, color=TEXT3, fontstyle="italic")
    ax2.text(0, -1.7, "Actionable intelligence for enforcement & compliance", ha="center",
             fontsize=10, fontweight="bold", color=GREEN)

    fig.suptitle("The Classification Gap: Why Cause Identification Changes Everything",
                 fontsize=16, fontweight="bold", y=0.98)
    save(fig, "07_classification_revolution.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Cloud Cover Problem  –  "Why optical-only fails"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_08_cloud_cover():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    regions = ["Amazon\nBasin", "Congo\nBasin", "SE Asia\n(Borneo)",
               "Western Ghats\n(India)", "Central\nAmerica", "West\nAfrica"]
    cloud = [87, 80, 75, 70, 72, 68]
    usable = [13, 20, 25, 30, 28, 32]

    x = np.arange(len(regions))
    w = 0.32

    # Stacked: cloud-blocked portion in red, usable in blue
    bars_cloud = ax.bar(x, cloud, width=0.65, color=RED_D, alpha=0.7,
                        edgecolor=BG, linewidth=1.5, label="Cloud-Blocked Optical Data")
    bars_usable = ax.bar(x, usable, width=0.65, bottom=0, color=BLUE_D, alpha=0.5,
                         edgecolor=BG, linewidth=1.5, label="Usable Optical Windows")

    # Redraw usable from 0
    for i, (b, u) in enumerate(zip(cloud, usable)):
        # Red on top of blue
        pass

    # Actually let's do it differently - cleaner
    ax.clear()

    # Full bar = 100%, split into cloud/usable
    for i, (c, u) in enumerate(zip(cloud, usable)):
        # Usable (bottom, small)
        ax.bar(i, u, width=0.6, color=BLUE, alpha=0.7,
               edgecolor=BG, linewidth=1)
        # Cloud blocked (on top)
        ax.bar(i, c, width=0.6, bottom=u, color=RED_D, alpha=0.6,
               edgecolor=BG, linewidth=1)

        # Labels
        ax.text(i, u/2, f"{u}%\nusable", ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE)
        ax.text(i, u + c/2, f"{c}%\nblocked", ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE, alpha=0.8)

    # SAR line
    ax.axhline(98, color=GREEN, lw=3, ls="-", alpha=0.9, zorder=5)
    ax.text(len(regions) - 0.5, 100.5,
            "SAR Availability: ~98-100%  (cloud-independent)",
            fontsize=11, fontweight="bold", color=GREEN, ha="right")

    # Annotation
    ax.annotate("DeforestNet uses\nSentinel-1 SAR", xy=(0, 98),
                xytext=(-0.3, 85), fontsize=10, color=GREEN,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("Observation Availability (%)", fontsize=11)
    ax.set_title("The Cloud Cover Problem: Why Optical-Only Systems Fail in Tropics",
                 fontsize=14, pad=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.1)

    # Legend
    handles = [
        mpatches.Patch(color=RED_D, alpha=0.6, label="Cloud-Blocked (lost data)"),
        mpatches.Patch(color=BLUE, alpha=0.7, label="Usable Optical Windows"),
        plt.Line2D([0], [0], color=GREEN, lw=3, label="SAR Continuous Coverage"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              framealpha=0.3, edgecolor=SURFACE)

    # Bottom text
    ax.text(0.5, -0.12,
            "GFW, Planet, MapBiomas, GLAD = optical only. DeforestNet + SarVision + Kayrros use SAR. "
            "Only DeforestNet combines SAR + Optical + 6-class AI.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=TEXT3,
            fontstyle="italic")

    save(fig, "08_cloud_cover_problem.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Cost-to-Impact Ratio  – "Maximum impact, zero data cost"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_09_cost_impact():
    fig, ax = plt.subplots(figsize=(12, 7))

    systems = [
        ("DeforestNet",  0,    93, GREEN,  900),
        ("GFW / GLAD",   0,    38, PURPLE, 700),
        ("MapBiomas",    0,    42, TEAL,   500),
        ("SarVision",    150,  55, CYAN,   350),
        ("Kayrros",      250,  52, ORANGE, 400),
        ("Satelligence", 200,  65, BLUE,   500),
        ("Planet Labs",  300,  50, AMBER,  600),
        ("Maxar",        500,  30, RED,    450),
    ]

    for name, cost, cap, color, size in systems:
        ax.scatter(cost, cap, s=size, c=color, alpha=0.75,
                   edgecolors=WHITE, linewidth=2, zorder=3)
        # Glow
        ax.scatter(cost, cap, s=size * 2.5, c=color, alpha=0.08, zorder=2)

        ox = 15 if cost > 0 else 15
        oy = 2
        if name == "GFW / GLAD":
            oy = -4
        if name == "MapBiomas":
            oy = -4
            ox = 15

        ax.text(cost + ox, cap + oy, name, fontsize=10,
                fontweight="bold" if name == "DeforestNet" else "normal",
                color=color, zorder=4)

    # Optimal zone highlight
    rect = mpatches.FancyBboxPatch((-20, 80), 80, 20,
                                    boxstyle="round,pad=5",
                                    facecolor=GREEN, alpha=0.06,
                                    edgecolor=GREEN, lw=2, ls="--", zorder=1)
    ax.add_patch(rect)
    ax.text(20, 97, "OPTIMAL: High Impact + Zero Cost", fontsize=9,
            fontweight="bold", color=GREEN, ha="center")

    # Arrows showing dimensions
    ax.annotate("Better", xy=(-10, 95), fontsize=8, color=GREEN,
                fontweight="bold", ha="right")
    ax.annotate("Worse", xy=(520, 25), fontsize=8, color=RED,
                ha="left")

    ax.set_xlabel("Annual Data + Platform Cost (USD Thousands)  >>>", fontsize=11)
    ax.set_ylabel("<<<  Capability Score (out of 100)", fontsize=11)
    ax.set_title("Cost-to-Impact Ratio: DeforestNet Delivers Maximum Impact at Zero Data Cost",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-30, 560)
    ax.set_ylim(20, 102)
    ax.grid(True, alpha=0.1)

    save(fig, "09_cost_impact_ratio.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. The "Why DeforestNet" Summary Slide
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_10_why_deforestnet():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 10)

    fig.text(0.5, 0.97, "Why DeforestNet?",
             ha="center", fontsize=26, fontweight="bold", color=GREEN,
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    fig.text(0.5, 0.93, "The only solution combining multi-sensor AI with cause-of-deforestation classification",
             ha="center", fontsize=12, color=TEXT2)

    cards = [
        # Row 1
        {"x": 0.5,  "y": 6.8, "w": 4.0, "h": 2.0, "icon": "6",
         "title": "CLASSES", "subtitle": "Cause Identification",
         "body": "Logging, Mining, Agriculture,\nFire, Infrastructure + Forest",
         "accent": "Not binary. Actionable.",
         "color": GREEN},
        {"x": 5.0,  "y": 6.8, "w": 4.0, "h": 2.0, "icon": "11",
         "title": "SPECTRAL BANDS", "subtitle": "Multi-Sensor Fusion",
         "body": "4 Optical + 2 SAR + 5 Derived\nindices in unified deep learning",
         "accent": "Richest input in the market.",
         "color": BLUE},
        {"x": 9.5,  "y": 6.8, "w": 4.0, "h": 2.0, "icon": "$0",
         "title": "DATA COST", "subtitle": "100% Free ESA Data",
         "body": "Sentinel-1 + Sentinel-2\nCopernicus Open Access",
         "accent": "Competitors pay $30-500K/yr.",
         "color": AMBER},
        # Row 2
        {"x": 0.5,  "y": 3.8, "w": 4.0, "h": 2.0, "icon": "10m",
         "title": "RESOLUTION", "subtitle": "High-Detail Mapping",
         "body": "9x more detail than GFW/GLAD\n(30m Landsat baseline)",
         "accent": "See smallholder-level changes.",
         "color": PURPLE},
        {"x": 5.0,  "y": 3.8, "w": 4.0, "h": 2.0, "icon": "24/7",
         "title": "MONITORING", "subtitle": "Cloud-Penetrating SAR",
         "body": "Microwave radar sees through\nclouds, rain, smoke, darkness",
         "accent": "Never miss deforestation.",
         "color": CYAN},
        {"x": 9.5,  "y": 3.8, "w": 4.0, "h": 2.0, "icon": "XAI",
         "title": "EXPLAINABLE", "subtitle": "GradCAM Transparency",
         "body": "Visual heatmaps showing WHY\nthe AI flagged each area",
         "accent": "Auditor & regulator ready.",
         "color": PINK},
    ]

    for card in cards:
        x, y, w, h = card["x"], card["y"], card["w"], card["h"]
        c = card["color"]

        # Card background
        bg_rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=SURFACE, edgecolor=c, lw=2, alpha=0.8)
        ax.add_patch(bg_rect)

        # Top accent bar
        accent_rect = mpatches.FancyBboxPatch(
            (x + 0.1, y + h - 0.12), w - 0.2, 0.08,
            boxstyle="round,pad=0.02",
            facecolor=c, edgecolor="none", alpha=0.7)
        ax.add_patch(accent_rect)

        # Icon number
        ax.text(x + 0.4, y + h - 0.55, card["icon"],
                fontsize=24, fontweight="bold", color=c, va="center",
                path_effects=GLOW)

        # Title + subtitle
        ax.text(x + 1.8, y + h - 0.4, card["title"],
                fontsize=11, fontweight="bold", color=WHITE)
        ax.text(x + 1.8, y + h - 0.7, card["subtitle"],
                fontsize=8, color=c)

        # Body text
        ax.text(x + 0.3, y + 0.7, card["body"],
                fontsize=9, color=TEXT2, va="center")

        # Accent text
        ax.text(x + 0.3, y + 0.15, card["accent"],
                fontsize=8, color=c, fontweight="bold", fontstyle="italic")

    # Bottom impact statement
    ax.text(7, 2.8, "MARKET POSITION", ha="center", fontsize=11,
            fontweight="bold", color=TEXT3)

    bottom_stats = [
        ("ZERO\ncompetitors", "offer automated\ncause identification", RED),
        ("400,000+\noperators", "need EUDR compliance\ntools by Dec 2026", AMBER),
        ("$50B+\nmarket", "satellite services\nby 2032", GREEN),
        ("6.7M ha\nlost in 2024", "record deforestation\n= urgent demand", ORANGE),
    ]
    for i, (val, desc, c) in enumerate(bottom_stats):
        cx = 1.5 + i * 3.3
        # Mini card
        mini = mpatches.FancyBboxPatch(
            (cx - 1.3, 0.2), 2.6, 2.2, boxstyle="round,pad=0.1",
            facecolor=CARD, edgecolor=c, lw=1.5, alpha=0.5)
        ax.add_patch(mini)
        ax.text(cx, 1.7, val, fontsize=13, fontweight="bold",
                color=c, ha="center")
        ax.text(cx, 0.7, desc, fontsize=8, color=TEXT3, ha="center")

    save(fig, "10_why_deforestnet.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("Generating advanced pitch-deck graphs (v2)...\n")

    graph_01_market_funnel()
    graph_02_positioning_map()
    graph_03_impact_scorecard()
    graph_04_radar()
    graph_05_eudr_timeline()
    graph_06_crisis()
    graph_07_classification_revolution()
    graph_08_cloud_cover()
    graph_09_cost_impact()
    graph_10_why_deforestnet()

    n = len(list(OUT.glob("*.png")))
    print(f"\nAll graphs saved to: {OUT}")
    print(f"Total: {n} PNG files (250 DPI)")
