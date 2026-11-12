"""Score every run in results/ and write REPORT.md, summary.json and charts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import costs
from .run import load
from .score import aggregate

SPLITS = ["test_id", "test_ood"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def collect(results_dir="results", data_dir="data", gpu_hourly=0.80, train_hours=1.5, batch=False,
            pricing: dict | None = None):
    """pricing maps run label -> 'api:<model>' or 'gpu'. Inferred from the run's spec if not given."""
    pricing = dict(pricing or {})
    gold = {s: {r["id"]: r["label"] for r in load(s, data_dir)} for s in SPLITS}
    models = {}
    for d in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        label = d.name
        m = {"label": label, "splits": {}}
        tin, tout, lat, dph, spec = [], [], [], [], ""
        for s in SPLITS:
            f = d / f"{s}.jsonl"
            if not f.exists():
                continue
            rows = [r for r in _rows(f) if r["id"] in gold[s]]
            agg = aggregate([gold[s][r["id"]] for r in rows], [r["pred"] for r in rows])
            agg["errors"] = sum(r["error"] is not None for r in rows)
            m["splits"][s] = agg
            tin += [r["input_tokens"] for r in rows if r["error"] is None]
            tout += [r["output_tokens"] for r in rows if r["error"] is None]
            lat += [r["latency_s"] for r in rows if r["error"] is None]
            meta = f.with_suffix(".meta.json")
            if meta.exists():
                mj = json.loads(meta.read_text())
                dph.append(mj["docs_per_hour"]); spec = mj.get("spec") or spec
        if not m["splits"]:
            continue
        m["mean_input_tokens"] = float(np.mean(tin)) if tin else 0
        m["mean_output_tokens"] = float(np.mean(tout)) if tout else 0
        m["latency_p50"] = float(np.percentile(lat, 50)) if lat else None
        m["latency_p95"] = float(np.percentile(lat, 95)) if lat else None
        m["docs_per_hour"] = float(np.mean(dph)) if dph else None
        kind = pricing.get(label) or (f"api:{spec.split(':', 1)[1]}" if spec.startswith("anthropic:") else
                                      "gpu" if spec.startswith("openai:") else None)
        m["pricing"] = kind
        if kind and kind.startswith("api:"):
            m["cost_per_1k"] = 1000 * costs.api_cost_per_doc(kind[4:], m["mean_input_tokens"],
                                                              m["mean_output_tokens"], batch)
        elif kind == "gpu" and m["docs_per_hour"]:
            m["cost_per_1k"] = 1000 * costs.selfhost_cost_per_doc(gpu_hourly, m["docs_per_hour"])
        else:
            m["cost_per_1k"] = None
        models[label] = m

    # break-even of every self-hosted model against every API model
    train_cost = gpu_hourly * train_hours
    be = []
    for g in [m for m in models.values() if m["pricing"] == "gpu" and m["docs_per_hour"]]:
        for a in [m for m in models.values() if (m["pricing"] or "").startswith("api:")]:
            v = costs.break_even_docs_per_day(a["cost_per_1k"] / 1000, gpu_hourly, g["docs_per_hour"], train_cost)
            be.append({"selfhost": g["label"], "api": a["label"], "docs_per_day": v})
    return {"models": models, "break_even": be,
            "assumptions": {"gpu_hourly_usd": gpu_hourly, "train_hours": train_hours,
                            "train_cost_usd": train_cost, "amortize_days": 30, "api_batch_pricing": batch,
                            "api_prices_per_mtok": costs.API_PRICES}}


def _fmt(x, kind="pct"):
    if x is None:
        return "—"
    return {"pct": f"{x:.1%}", "usd": f"${x:,.2f}" if x >= 0.1 else f"${x:,.3f}",
            "s": f"{x:.2f}s", "int": f"{x:,.0f}"}[kind]


def markdown(res) -> str:
    M, A = res["models"], res["assumptions"]
    L = ["# Fine-tuned small model vs frontier API: clinical note extraction", "",
         "## Accuracy", "",
         "| model | exact match (ID) | exact match (OOD) | field acc (ID) | field acc (OOD) | valid JSON (OOD) |",
         "|---|---|---|---|---|---|"]
    for m in M.values():
        s = m["splits"]
        g = lambda sp, k: s[sp][k] if sp in s else None
        L.append(f"| {m['label']} | {_fmt(g('test_id', 'exact_match'))} | {_fmt(g('test_ood', 'exact_match'))} | "
                 f"{_fmt(g('test_id', 'mean_field_accuracy'))} | {_fmt(g('test_ood', 'mean_field_accuracy'))} | "
                 f"{_fmt(g('test_ood', 'valid_json'))} |")
    L += ["", "ID = same note styles as training, new patients. OOD = new note styles, Latin abbreviations, "
              "Celsius, follow-up in days, and conditions/drugs never seen in training.", "",
          "## Generalisation: recall on seen vs never-seen vocabulary (OOD split)", "",
          "| model | ICD-10 seen | ICD-10 held-out | meds seen | meds held-out | false positives / note |",
          "|---|---|---|---|---|---|"]
    for m in M.values():
        o = m["splits"].get("test_ood")
        if not o:
            continue
        r = o["recall_by_vocab"]
        g = lambda k: r[k]["recall"] if k in r else None
        L.append(f"| {m['label']} | {_fmt(g('icd10_seen'))} | {_fmt(g('icd10_heldout'))} | {_fmt(g('meds_seen'))} | "
                 f"{_fmt(g('meds_heldout'))} | {o['false_positive_items_per_doc']:.2f} |")
    L += ["", "## Cost and speed", "",
          "| model | pricing | cost / 1,000 notes | cost / 1,000 *correct* notes (OOD) | mean tokens in / out | latency p50 / p95 | notes / hour |",
          "|---|---|---|---|---|---|---|"]
    for m in M.values():
        em = m["splits"].get("test_ood", {}).get("exact_match")
        cpc = m["cost_per_1k"] / em if m["cost_per_1k"] is not None and em else None
        L.append(f"| {m['label']} | {m['pricing'] or '—'} | {_fmt(m['cost_per_1k'], 'usd')} | {_fmt(cpc, 'usd')} | "
                 f"{m['mean_input_tokens']:.0f} / {m['mean_output_tokens']:.0f} | "
                 f"{_fmt(m['latency_p50'], 's')} / {_fmt(m['latency_p95'], 's')} | {_fmt(m['docs_per_hour'], 'int')} |")
    L += ["", "## When does self-hosting beat the API?", ""]
    for b in res["break_even"]:
        v = b["docs_per_day"]
        L.append(f"- **{b['selfhost']}** vs **{b['api']}**: " +
                 (f"self-hosting is cheaper above **{v:,.0f} notes/day**" if v else
                  "never: a fully busy GPU still costs more per note than the API"))
    L += ["", "Assumptions: one always-on GPU at "
              f"${A['gpu_hourly_usd']:.2f}/hour (set with --gpu-hourly), training {A['train_hours']} GPU-hours "
              f"amortised over {A['amortize_days']} days, API list prices"
              f"{' with 50% Batch API discount' if A['api_batch_pricing'] else ''}. "
              "Self-hosted throughput is what vLLM measured at the concurrency used in the run.", ""]
    return "\n".join(L)


def charts(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = list(res["models"].values())
    out = Path(out_dir)

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

    # 1. exact match, in-distribution vs out-of-distribution
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    style(ax, "Exact-match accuracy: familiar notes vs unfamiliar notes")
    x = np.arange(len(M)); w = 0.38
    for k, (sp, name) in enumerate([("test_id", "Familiar styles (ID)"), ("test_ood", "New styles + unseen terms (OOD)")]):
        vals = [m["splits"].get(sp, {}).get("exact_match", 0) for m in M]
        bars = ax.bar(x + (k - 0.5) * w, vals, w * 0.95, color=COLORS[k], label=name)
        for b, v in zip(bars, vals):
            lab = f"{v:.1%}" if 0.995 <= v < 1 else f"{v:.0%}"   # never round 99.7% up to "100%"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, lab, ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [m["label"] for m in M]); ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "accuracy.png", dpi=160); plt.close(fig)

    # 2. daily cost vs volume, with break-even points
    priced = [m for m in M if m["cost_per_1k"] is not None]
    if not priced:
        return
    A = res["assumptions"]
    vol = np.logspace(2, 7, 400)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    style(ax, f"Cost per day vs notes per day (GPU at ${A['gpu_hourly_usd']:.2f}/hour)")
    # self-hosted models with ~the same throughput have the same cost line: draw it once
    gpu_groups = {}
    for m in priced:
        if m["pricing"] == "gpu":
            gpu_groups.setdefault(round(m["docs_per_hour"], -2), []).append(m)
    lines = [(" / ".join(x["label"] for x in g) + " (self-hosted)", "gpu", g[0]) for g in gpu_groups.values()]
    lines += [(f"{m['label']} (API)", "api", m) for m in priced if m["pricing"] != "gpu"]
    for k, (name, kind, m) in enumerate(lines):
        if kind == "gpu":
            y = [costs.selfhost_cost_per_day(v, A["gpu_hourly_usd"], m["docs_per_hour"], A["train_cost_usd"])
                 for v in vol]
            ax.plot(vol, y, color=COLORS[k % 5], lw=2, drawstyle="steps-post", label=name)
        else:
            ax.plot(vol, vol * m["cost_per_1k"] / 1000, color=COLORS[k % 5], lw=2, label=name)
    for b in res["break_even"]:
        if b["docs_per_day"]:
            a = res["models"][b["api"]]
            v = b["docs_per_day"]
            ax.scatter([v], [v * a["cost_per_1k"] / 1000], s=60, color=INK, zorder=5)
            ax.annotate(f"{v:,.0f}/day", (v, v * a["cost_per_1k"] / 1000), xytext=(6, -14),
                        textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Notes per day", color=INK2); ax.set_ylabel("USD per day", color=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:g}"))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "break_even.png", dpi=160); plt.close(fig)


def write(results_dir="results", data_dir="data", **kw):
    res = collect(results_dir, data_dir, **kw)
    Path(results_dir, "summary.json").write_text(json.dumps(res, indent=2, default=str))
    md = markdown(res)
    Path(results_dir, "REPORT.md").write_text(md)
    charts(res, results_dir)
    return md
