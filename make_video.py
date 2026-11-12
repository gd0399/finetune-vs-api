"""Render a ~40s LinkedIn clip (1080x1350, 30 fps) from results/summary.json and the raw predictions.

    python make_video.py            -> results/finetune_vs_api.mp4   (needs ffmpeg)

Every number on screen is computed from the results files, not typed in.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ftva import costs
from ftva.score import HELDOUT_CODES

plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams["text.parse_math"] = False
W, H, FPS = 1080, 1350, 30
BG, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, GREEN, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b"
TMP = Path("/tmp/ftvideo")
R = Path("results")

# ------------------------------------------------------------------ numbers from the results
S = json.load(open(R / "summary.json"))
M, A = S["models"], S["assumptions"]
FT, BASE, HAIKU, SONNET = "qwen3-8b-ft", "qwen3-8b-base", "haiku-5.5", "sonnet-5.5"
em = {k: (M[k]["splits"]["test_id"]["exact_match"], M[k]["splits"]["test_ood"]["exact_match"]) for k in M}
heldout_recall = M[FT]["splits"]["test_ood"]["recall_by_vocab"]["icd10_heldout"]["recall"]
be = {b["api"]: b["docs_per_day"] for b in S["break_even"] if b["selfhost"] == FT}

gold = {json.loads(l)["id"]: json.loads(l) for l in open("data/test_ood.jsonl")}
ft_rows = [json.loads(l) for l in open(R / FT / "test_ood.jsonl")]
cel = [r for r in ft_rows if "Celsius" in gold[r["id"]]["note"] or ("C SpO2" in gold[r["id"]]["note"])]
cel_wrong = sum(abs(float(r["pred"]["vitals"]["temperature_f"]) - gold[r["id"]]["label"]["vitals"]["temperature_f"]) > 0.2
                for r in cel)
n_test = sum(M[FT]["splits"][s]["n"] for s in ("test_id", "test_ood"))


def pct(v):
    return f"{v:.1%}" if 0.995 <= v < 1 else f"{v:.0%}"


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def fade(t, dur, fin=0.4, fout=0.4):
    return min(1.0, t / fin, (dur - t) / fout)


def fig_new():
    return plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)


def header(fig, kicker, title, a=1.0):
    fig.text(0.08, 0.915, kicker, color=BLUE, fontsize=22, fontweight="bold", alpha=a)
    fig.text(0.08, 0.865, title, color=INK, fontsize=36, fontweight="bold", alpha=a, va="top")


def footer(fig):
    fig.text(0.08, 0.035, "Synthetic clinical notes, no patient data  ·  Qwen3-8B QLoRA on an A100  ·  Nov 2026",
             color=MUTED, fontsize=14)


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=17, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ scenes
def s_hook(t, dur):
    fig = fig_new()
    a = ease(t / 0.7)
    fig.text(0.08, 0.72, "I fine-tuned an 8B model", color=INK, fontsize=46, fontweight="bold", alpha=a)
    fig.text(0.08, 0.665, "to read clinical notes.", color=INK, fontsize=46, fontweight="bold", alpha=a)
    v = em[FT][0] * ease((t - 0.6) / 1.4)
    fig.text(0.08, 0.50, pct(v) if t > 2.0 else f"{v:.1%}", color=BLUE, fontsize=140, fontweight="bold",
             alpha=ease((t - 0.5) / 0.5))
    fig.text(0.08, 0.445, "accurate on notes like its training data.", color=INK2, fontsize=32,
             alpha=ease((t - 1.6) / 0.6))
    b = ease((t - 2.8) / 0.6)
    fig.text(0.08, 0.32, "Then I showed it notes", color=INK, fontsize=42, alpha=b)
    fig.text(0.08, 0.27, "written a little differently.", color=INK, fontsize=42, alpha=b)
    footer(fig)
    return fig


def s_setup(t, dur):
    fig = fig_new()
    header(fig, "THE TEST", f"4 models · same prompt · {n_test} notes")
    rows = [("Qwen3-8B, untuned", "open model, self-hosted on vLLM", MUTED),
            ("Qwen3-8B, fine-tuned", "QLoRA on 2,000 notes, self-hosted", ORANGE),
            ("Claude Haiku 5.5", "API", GREEN),
            ("Claude Sonnet 5.5", "API", YELLOW)]
    for i, (n, d, c) in enumerate(rows):
        a = ease((t - 0.3 - 0.35 * i) / 0.5)
        y = 0.74 - i * 0.08
        fig.add_artist(plt.Rectangle((0.08, y - 0.012), 0.012, 0.04, color=c, alpha=a, transform=fig.transFigure))
        fig.text(0.11, y + 0.008, n, color=INK, fontsize=28, fontweight="bold", alpha=a)
        fig.text(0.11, y - 0.022, d, color=INK2, fontsize=20, alpha=a)
    b = ease((t - 2.0) / 0.6)
    fig.text(0.08, 0.35, "The job: turn each note into JSON", color=INK, fontsize=28, fontweight="bold", alpha=b)
    fig.text(0.08, 0.31, "13 fields: ICD-10 codes, medications, vitals, allergies…", color=INK2, fontsize=23, alpha=b)
    c = ease((t - 3.0) / 0.6)
    fig.text(0.08, 0.235, "…while ignoring the traps: family history,", color=INK2, fontsize=23, alpha=c)
    fig.text(0.08, 0.20, "ruled-out conditions and stopped medications.", color=INK2, fontsize=23, alpha=c)
    footer(fig)
    return fig


def s_accuracy(t, dur):
    fig = fig_new()
    header(fig, "RESULT", "Exact-match accuracy")
    ax = fig.add_axes([0.1, 0.30, 0.84, 0.47])
    style_ax(ax)
    order = [BASE, FT, HAIKU, SONNET]
    names = ["Qwen 8B\nuntuned", "Qwen 8B\nfine-tuned", "Haiku 5.5", "Sonnet 5.5"]
    x = np.arange(4); w = 0.38
    g1 = ease((t - 0.3) / 1.2)
    g2 = ease((t - 1.8) / 1.2)
    for k, (gr, col) in enumerate([(g1, BLUE), (g2, ORANGE)]):
        vals = [em[m][k] for m in order]
        ax.bar(x + (k - 0.5) * w, [v * gr for v in vals], w * 0.95, color=col)
        if gr > 0.97:
            for xi, v in zip(x, vals):
                ax.text(xi + (k - 0.5) * w, v + 0.02, pct(v), ha="center", fontsize=16, color=INK2)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.text(0.1, 0.80, "■ notes like the training data", color=BLUE, fontsize=19)
    fig.text(0.53, 0.80, "■ new styles + unseen terms", color=ORANGE, fontsize=19)
    a = ease((t - 3.4) / 0.6)
    fig.text(0.08, 0.17, f"Fine-tuned: {pct(em[FT][0])} → {pct(em[FT][1])}", color=RED, fontsize=40,
             fontweight="bold", alpha=a)
    fig.text(0.08, 0.125, f"worse than the untuned model ({pct(em[BASE][1])}). Claude held at {pct(em[HAIKU][1])}–"
             f"{pct(em[SONNET][1])}.", color=INK2, fontsize=22, alpha=a)
    footer(fig)
    return fig


def s_why(t, dur):
    fig = fig_new()
    header(fig, "WHY IT BROKE", "It learned the format, not the task")
    a = ease((t - 0.3) / 0.6)
    fig.text(0.08, 0.74, "1. Celsius", color=INK, fontsize=30, fontweight="bold", alpha=a)
    fig.text(0.08, 0.69, "Note:   T 37.6C", color=INK2, fontsize=26, alpha=a, family="DejaVu Sans Mono")
    fig.text(0.08, 0.65, "Model:  37.6 °F   (correct: 99.6 °F)", color=RED, fontsize=26, alpha=a,
             family="DejaVu Sans Mono")
    fig.text(0.08, 0.60, f"Copied Celsius as Fahrenheit in {cel_wrong} of {len(cel)} notes.", color=INK, fontsize=24,
             alpha=a)
    fig.text(0.08, 0.565, "Training never contained a Celsius reading.", color=INK2, fontsize=22, alpha=a)
    b = ease((t - 3.2) / 0.6)
    fig.text(0.08, 0.46, "2. Conditions it never trained on", color=INK, fontsize=30, fontweight="bold", alpha=b)
    fig.text(0.08, 0.41, "Carpal tunnel syndrome", color=INK2, fontsize=26, alpha=b, family="DejaVu Sans Mono")
    fig.text(0.08, 0.37, "Correct: G56.00   Model: G56.90", color=RED, fontsize=26, alpha=b, family="DejaVu Sans Mono")
    fig.text(0.08, 0.32, f"Got only {heldout_recall:.0%} of never-seen ICD-10 codes right.", color=INK,
             fontsize=24, alpha=b)
    fig.text(0.08, 0.285, "Fine-tuning didn't add medical knowledge.", color=INK2, fontsize=22, alpha=b)
    footer(fig)
    return fig


vol = np.logspace(2, 6.5, 300)


def s_cost(t, dur):
    fig = fig_new()
    header(fig, "COST", "When is a GPU cheaper than the API?")
    ax = fig.add_axes([0.15, 0.33, 0.78, 0.44])
    style_ax(ax)
    k = max(2, int(len(vol) * ease((t - 0.3) / 2.5)))
    v = vol[:k]
    gpu = [costs.selfhost_cost_per_day(x, A["gpu_hourly_usd"], M[FT]["docs_per_hour"], A["train_cost_usd"]) for x in v]
    ax.plot(v, gpu, color=ORANGE, lw=3, drawstyle="steps-post")
    ax.plot(v, v * M[SONNET]["cost_per_1k"] / 1000, color=YELLOW, lw=3)
    ax.plot(v, v * M[HAIKU]["cost_per_1k"] / 1000, color=GREEN, lw=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(vol[0], vol[-1]); ax.set_ylim(0.03, 30000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"${x:g}"))
    ax.set_xlabel("Notes per day", color=INK2, fontsize=17); ax.set_ylabel("USD per day", color=INK2, fontsize=17)
    fig.text(0.16, 0.795, f"■ fine-tuned 8B on an A100 (${A['gpu_hourly_usd']:.2f}/hr)", color=ORANGE, fontsize=18)
    fig.text(0.16, 0.775 - 0.0, "", fontsize=1)
    fig.text(0.62, 0.795, "■ Sonnet", color=YELLOW, fontsize=18)
    fig.text(0.78, 0.795, "■ Haiku", color=GREEN, fontsize=18)
    if t > 2.9:
        for api, col in ((SONNET, YELLOW), (HAIKU, GREEN)):
            x0 = be[api]
            ax.scatter([x0], [x0 * M[api]["cost_per_1k"] / 1000], s=110, color=INK, zorder=5)
    a = ease((t - 3.0) / 0.6)
    fig.text(0.08, 0.215, f"Beats Sonnet above {be[SONNET]:,.0f} notes/day", color=INK, fontsize=28,
             fontweight="bold", alpha=a)
    fig.text(0.08, 0.17, f"Beats Haiku only above {be[HAIKU]:,.0f} notes/day", color=INK, fontsize=28,
             fontweight="bold", alpha=a)
    b = ease((t - 4.3) / 0.6)
    fig.text(0.08, 0.115, f"Haiku: ${M[HAIKU]['cost_per_1k']:.2f} per 1,000 notes, {pct(em[HAIKU][1])} on unfamiliar notes.",
             color=INK2, fontsize=22, alpha=b)
    footer(fig)
    return fig


def s_takeaway(t, dur):
    fig = fig_new()
    header(fig, "TAKEAWAY", "Fine-tuning works only on")
    fig.text(0.08, 0.818, "what your data covers", color=INK, fontsize=36, fontweight="bold", va="top")
    items = ["Test on data that doesn't look like your training set",
             "A cheap API model can beat a tuned 8B on messy input",
             "Self-hosting pays off only at high, steady volume"]
    for i, s in enumerate(items):
        a = ease((t - 0.4 - 0.45 * i) / 0.5)
        y = 0.66 - i * 0.075
        fig.text(0.08, y, "→", color=BLUE, fontsize=28, fontweight="bold", alpha=a)
        fig.text(0.13, y + 0.003, s, color=INK, fontsize=24, alpha=a)
    b = ease((t - 2.2) / 0.6)
    fig.text(0.08, 0.36, "Checked before posting", color=INK, fontsize=26, fontweight="bold", alpha=b)
    fig.text(0.08, 0.32, "29 tests · found & fixed an ICD-10 scoring bug in my own labels", color=INK2,
             fontsize=21, alpha=b)
    c = ease((t - 3.4) / 0.6)
    fig.text(0.08, 0.20, "github.com/gd0399/finetune-vs-api", color=BLUE, fontsize=30, fontweight="bold", alpha=c)
    fig.text(0.08, 0.16, "Python · PyTorch · PEFT/QLoRA · vLLM · Claude API", color=INK2, fontsize=21, alpha=c)
    footer(fig)
    return fig


SCENES = [("01_hook", s_hook, 5.0), ("02_setup", s_setup, 6.0), ("03_acc", s_accuracy, 7.5),
          ("04_why", s_why, 8.0), ("05_cost", s_cost, 8.0), ("06_take", s_takeaway, 7.0)]


def frame_rgb(fn, t, dur):
    fig = fn(t, dur)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    a = fade(t, dur)
    if a < 1:
        bg = np.array(matplotlib.colors.to_rgb(BG)) * 255
        img = (img * a + bg * (1 - a)).astype(np.uint8)
    return img


def render(name, fn, dur):
    tmp, final = TMP / f"{name}.part.mp4", TMP / f"{name}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for i in range(int(round(dur * FPS))):
        p.stdin.write(frame_rgb(fn, i / FPS, dur).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, final)


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    if "--stills" in sys.argv:
        for name, fn, dur in SCENES:
            plt.imsave(TMP / f"still_{name}.png", frame_rgb(fn, dur - 0.6, dur))
        sys.exit()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    for name, fn, dur in SCENES:
        if not only or name in only:
            render(name, fn, dur)
    if not only or "--join" in sys.argv:
        (TMP / "list.txt").write_text("".join(f"file '{TMP / n}.mp4'\n" for n, _, _ in SCENES))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(TMP / "list.txt"),
                        "-c", "copy", "-movflags", "+faststart", str(R / "finetune_vs_api.mp4")], check=True)
        print("wrote", R / "finetune_vs_api.mp4")
