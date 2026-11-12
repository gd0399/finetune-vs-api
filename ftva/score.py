"""Field-level scoring of predictions against gold labels.

Per document we score 13 fields. Scalars are right/wrong; list fields get
precision/recall/F1 (micro-averaged over the split). A document is an
EXACT MATCH only if every field is right.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .vocab import HELDOUT_CONDITIONS, HELDOUT_MEDICATIONS

HELDOUT_CODES = {c for _, c, _ in HELDOUT_CONDITIONS}
HELDOUT_MEDS = {n for n, _, _ in HELDOUT_MEDICATIONS}
SCALARS = ["age", "sex", "chief_complaint", "smoking_status", "follow_up_weeks",
           "bp_systolic", "bp_diastolic", "heart_rate", "temperature_f", "spo2"]
LISTS = ["diagnoses", "medications", "allergies"]
TEMP_TOL = 0.2      # Celsius notes round-trip with up to 0.1 F rounding error


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", str(s).lower()).strip()


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _scalar_ok(field, gold, pred) -> bool:
    if field == "temperature_f":
        p = _num(pred)
        return p is not None and abs(p - gold) <= TEMP_TOL
    if field == "follow_up_weeks":
        if gold is None:
            return pred in (None, "null", "")
        return _num(pred) == gold
    if isinstance(gold, int):
        return _num(pred) == gold
    return _norm(gold) == _norm(pred)


def _get(d, field):
    if not isinstance(d, dict):
        return None
    if field in ("bp_systolic", "bp_diastolic", "heart_rate", "temperature_f", "spo2"):
        v = d.get("vitals")
        return v.get(field) if isinstance(v, dict) else None
    return d.get(field)


def _items(field, obj):
    """Turn a list field into a set of comparable keys."""
    xs = obj.get(field) if isinstance(obj, dict) else None
    if not isinstance(xs, list):
        return set()
    out = set()
    for x in xs:
        if field == "diagnoses" and isinstance(x, dict):
            out.add(str(x.get("icd10", "")).upper().strip())
        elif field == "medications" and isinstance(x, dict):
            out.add((_norm(x.get("name")), _norm(x.get("dose")).replace(" ", ""), _norm(x.get("frequency"))))
        elif field == "allergies":
            out.add(_norm(x))
    return out


def score_doc(gold: dict, pred: dict | None) -> dict:
    """Return per-field results for one document."""
    r = {"valid_json": pred is not None, "fields": {}, "lists": {}}
    for f in SCALARS:
        r["fields"][f] = pred is not None and _scalar_ok(f, _get(gold, f), _get(pred, f))
    for f in LISTS:
        g, p = _items(f, gold), _items(f, pred or {})
        r["lists"][f] = {"tp": len(g & p), "fp": len(p - g), "fn": len(g - p),
                         "exact": g == p, "gold": g, "pred": p}
        r["fields"][f] = g == p
    r["exact_match"] = all(r["fields"].values())
    return r


def aggregate(golds: list[dict], preds: list[dict | None]) -> dict:
    docs = [score_doc(g, p) for g, p in zip(golds, preds)]
    n = len(docs)
    out = {"n": n,
           "valid_json": sum(d["valid_json"] for d in docs) / n,
           "exact_match": sum(d["exact_match"] for d in docs) / n,
           "field_accuracy": {}, "list_f1": {}}
    for f in SCALARS + LISTS:
        out["field_accuracy"][f] = sum(d["fields"][f] for d in docs) / n
    out["mean_field_accuracy"] = sum(out["field_accuracy"].values()) / len(out["field_accuracy"])
    for f in LISTS:
        tp = sum(d["lists"][f]["tp"] for d in docs)
        fp = sum(d["lists"][f]["fp"] for d in docs)
        fn = sum(d["lists"][f]["fn"] for d in docs)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        out["list_f1"][f] = {"precision": prec, "recall": rec,
                             "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    # seen vs held-out vocabulary: recall on gold items only
    rec = defaultdict(lambda: [0, 0])
    for d in docs:
        for code in d["lists"]["diagnoses"]["gold"]:
            k = "icd10_heldout" if code in HELDOUT_CODES else "icd10_seen"
            rec[k][1] += 1
            rec[k][0] += code in d["lists"]["diagnoses"]["pred"]
        for m in d["lists"]["medications"]["gold"]:
            k = "meds_heldout" if m[0] in HELDOUT_MEDS else "meds_seen"
            rec[k][1] += 1
            rec[k][0] += m in d["lists"]["medications"]["pred"]
    out["recall_by_vocab"] = {k: {"recall": a / b, "n": b} for k, (a, b) in rec.items()}
    # distractors: anything predicted that isn't gold is a false positive; count them per doc
    out["false_positive_items_per_doc"] = sum(
        d["lists"][f]["fp"] for d in docs for f in ("diagnoses", "medications")) / n
    return out
