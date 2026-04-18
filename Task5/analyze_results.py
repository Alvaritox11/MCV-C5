"""
Diffusion Exploration — Result Analyser
========================================
After `explore_diffusion.py` has run, call this script to generate:
  - Side-by-side comparison grids (per experiment, per model)
  - A summary HTML report
  - Timing bar charts
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def load_meta(path: Path) -> Optional[dict]:
    meta_file = path.with_suffix(".json")
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)
    return {}


def label_image(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Add a bottom label strip to a PIL image."""
    bar_h = font_size + 10
    out = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, img.height + 4), text, fill=(220, 220, 220), font=font)
    return out


def make_grid(images: list[tuple[Image.Image, str]], cols: int = 4,
              cell_size: int = 300) -> Image.Image:
    """Arrange labelled images in a grid."""
    rows = (len(images) + cols - 1) // cols
    label_h = 28
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_h)
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    for idx, (img, lbl) in enumerate(images):
        col = idx % cols
        row = idx // cols
        thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
        thumb = label_image(thumb, lbl, font_size=14)
        grid.paste(thumb, (col * cell_size, row * (cell_size + label_h)))
    return grid


# ─────────────────────────────────────────────────────────────────────
# Per-experiment grid builders
# ─────────────────────────────────────────────────────────────────────
EXP_DIRS = {
    "scheduler": "1_scheduler_comparison",
    "prompting": "2_prompting",
    "cfg": "3_cfg_sweep",
    "steps": "4_steps_sweep",
}


def build_experiment_grid(model_dir: Path, exp_key: str, out_dir: Path):
    exp_dir = model_dir / EXP_DIRS[exp_key]
    if not exp_dir.exists():
        return

    pngs = sorted(exp_dir.glob("*.png"))
    if not pngs:
        return

    cells = []
    for p in pngs:
        meta = load_meta(p)
        img = Image.open(p).convert("RGB")

        # Build a concise label
        if exp_key == "scheduler":
            lbl = f"sched={meta.get('scheduler', p.stem)}"
        elif exp_key == "prompting":
            lbl = meta.get("variant", p.stem).replace("_", "\n")
        elif exp_key == "cfg":
            lbl = f"CFG={meta.get('cfg', '?')}"
        elif exp_key == "steps":
            lbl = f"steps={meta.get('steps', '?')}  ({meta.get('elapsed_s', '?')}s)"
        else:
            lbl = p.stem

        cells.append((img, lbl))

    cols = min(len(cells), 4)
    grid = make_grid(cells, cols=cols)
    model_name = model_dir.name
    out_path = out_dir / f"{model_name}_{exp_key}_grid.png"
    grid.save(out_path)
    print(f"  Grid saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────
# Timing chart
# ─────────────────────────────────────────────────────────────────────
def build_timing_chart(output_root: Path, out_dir: Path):
    """Collect step-sweep timings and plot a grouped bar chart."""
    model_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    data = {}   # model_name -> {steps: elapsed_s}

    for md in model_dirs:
        steps_dir = md / EXP_DIRS["steps"]
        if not steps_dir.exists():
            continue
        timings = {}
        for p in sorted(steps_dir.glob("*.png")):
            meta = load_meta(p)
            s = meta.get("steps")
            t = meta.get("elapsed_s")
            if s is not None and t is not None:
                timings[s] = t
        if timings:
            data[md.name] = timings

    if not data:
        return

    all_steps = sorted({s for v in data.values() for s in v})
    x = range(len(all_steps))
    n_models = len(data)
    bar_w = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, len(all_steps) * 2), 5))
    colors = plt.cm.tab10.colors

    for i, (model_name, timings) in enumerate(data.items()):
        offsets = [xi + i * bar_w - (n_models - 1) * bar_w / 2 for xi in x]
        heights = [timings.get(s, 0) for s in all_steps]
        ax.bar(offsets, heights, width=bar_w, label=model_name, color=colors[i % 10])

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in all_steps])
    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Elapsed (s)")
    ax.set_title("Inference Time vs. Denoising Steps per Model")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "timing_chart.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Timing chart → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────
def build_html_report(output_root: Path, grids_dir: Path):
    model_dirs = sorted(p for p in output_root.iterdir() if p.is_dir() and p.name != "grids")

    sections = []
    for md in model_dirs:
        imgs_html = []
        for exp_key in EXP_DIRS:
            grid_file = grids_dir / f"{md.name}_{exp_key}_grid.png"
            if grid_file.exists():
                rel = grid_file.relative_to(output_root)
                imgs_html.append(
                    f'<figure style="margin:0">'
                    f'<img src="{rel}" style="width:100%;border-radius:6px">'
                    f'<figcaption style="color:#aaa;font-size:.85rem;margin-top:4px">'
                    f'{exp_key.upper()} experiment</figcaption>'
                    f'</figure>'
                )
        if imgs_html:
            sections.append(f"""
<section>
  <h2 style="border-bottom:2px solid #444;padding-bottom:8px;color:#e0e0ff">{md.name}</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px">
    {''.join(imgs_html)}
  </div>
</section>""")

    # timing chart
    tc = grids_dir / "timing_chart.png"
    timing_html = ""
    if tc.exists():
        rel = tc.relative_to(output_root)
        timing_html = f"""
<section>
  <h2 style="border-bottom:2px solid #444;padding-bottom:8px;color:#e0e0ff">Timing Comparison</h2>
  <img src="{rel}" style="max-width:900px;width:100%;border-radius:6px">
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diffusion Exploration Report</title>
<style>
  body {{font-family:system-ui,sans-serif;background:#1a1a2e;color:#ccc;max-width:1200px;margin:auto;padding:24px}}
  h1 {{color:#a9c4f5;letter-spacing:1px}}
  figcaption {{text-align:center}}
</style>
</head>
<body>
<h1>🔬 Diffusion Model Exploration Report</h1>
<p style="color:#888">Generated automatically by <code>analyse_results.py</code></p>
{timing_html}
{''.join(sections)}
</body>
</html>"""

    report_path = output_root / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"  HTML report → {report_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main(output_root: str = "outputs"):
    root = Path(output_root)
    if not root.exists():
        print(f"Output directory '{root}' not found. Run explore_diffusion.py first.")
        return

    grids_dir = root / "grids"
    grids_dir.mkdir(exist_ok=True)

    model_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != "grids")

    for md in model_dirs:
        print(f"\nProcessing model: {md.name}")
        for exp_key in EXP_DIRS:
            build_experiment_grid(md, exp_key, grids_dir)

    print("\nBuilding timing chart ...")
    build_timing_chart(root, grids_dir)

    print("Building HTML report ...")
    build_html_report(root, grids_dir)

    print(f"\n✅ Analysis complete. Open: {(root / 'report.html').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs",
                        help="Root output dir used by explore_diffusion.py")
    args = parser.parse_args()
    main(args.output)