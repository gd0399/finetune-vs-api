import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftva import costs  # noqa: E402
from ftva.generate import make_split, sample_record, SPOKEN  # noqa: E402
from ftva.prompt import parse, target  # noqa: E402
from ftva.score import aggregate, score_doc  # noqa: E402
from ftva.vocab import CONDITIONS, HELDOUT_CONDITIONS, HELDOUT_MEDICATIONS, MEDICATIONS  # noqa: E402

HELD_CODES = {c for _, c, _ in HELDOUT_CONDITIONS}
HELD_MEDS = {n for n, _, _ in HELDOUT_MEDICATIONS}


# ---------------- data generator
def test_deterministic():
    assert make_split(20, 7, False) == make_split(20, 7, False)


def test_every_condition_has_spoken_forms():
    for _, code, _ in CONDITIONS + HELDOUT_CONDITIONS:
        assert code in SPOKEN


def test_heldout_vocab_never_in_training_splits():
    for r in make_split(1500, 1, False):
        assert not {d["icd10"] for d in r["label"]["diagnoses"]} & HELD_CODES
        assert not {m["name"] for m in r["label"]["medications"]} & HELD_MEDS


def test_ood_split_contains_heldout_vocab():
    rows = make_split(300, 4, True)
    assert sum(any(d["icd10"] in HELD_CODES for d in r["label"]["diagnoses"]) for r in rows) > 100


def test_ood_styles_differ_from_training_styles():
    assert {r["style"] for r in make_split(200, 1, False)} == {"soap", "narrative", "hp"}
    assert {r["style"] for r in make_split(200, 4, True)} == {"telegraphic", "dictation"}


def test_distractors_are_not_in_labels():
    import random
    rng = random.Random(3)
    for _ in range(2000):
        r = sample_record(rng, heldout=True)
        codes = {d["icd10"] for d in r["label"]["diagnoses"]}
        meds = {m["name"] for m in r["label"]["medications"]}
        if r["meta"]["family_history"]:
            assert r["meta"]["family_history"][1] not in codes
        if r["meta"]["ruled_out"]:
            assert r["meta"]["ruled_out"][1] not in codes
        if r["meta"]["discontinued"]:
            assert r["meta"]["discontinued"][0] not in meds


def test_everything_in_the_label_is_written_in_the_note():
    """Each gold medication, vital and age must actually appear in the note text."""
    for ood, seed in ((False, 1), (True, 4)):
        for r in make_split(300, seed, ood):
            note, L = r["note"].lower(), r["label"]
            assert str(L["age"]) in note
            for m in L["medications"]:
                assert m["name"] in note
            v = L["vitals"]
            for k in ("bp_systolic", "bp_diastolic", "heart_rate", "spo2"):
                assert str(v[k]) in note
            for a in L["allergies"]:
                assert a in note


def test_celsius_round_trip_within_tolerance():
    for f10 in range(970, 1009):
        f = f10 / 10
        c = round((f - 32) * 5 / 9, 1)
        assert abs((c * 9 / 5 + 32) - f) <= 0.2


# ---------------- prompt parsing
@pytest.mark.parametrize("text,ok", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),
    ('<think>hmm {"x":0}</think>{"a": 1}', True),
    ('Here you go: {"a": "say \\"hi\\" {not a brace}"} done', True),
    ("no json here", False),
    ('{"a": 1', False),
    (None, False),
])
def test_parse(text, ok):
    p = parse(text)
    assert (p is not None) == ok
    if ok:
        assert "a" in p


# ---------------- scoring
def _gold():
    return make_split(1, 9, False)[0]["label"]


def test_perfect_prediction_is_exact_match():
    g = _gold()
    assert score_doc(g, json.loads(target(g)))["exact_match"]


def test_extra_medication_is_a_false_positive():
    g = _gold()
    p = json.loads(target(g))
    p["medications"].append({"name": "apixaban", "dose": "5 mg", "frequency": "twice daily"})
    r = score_doc(g, p)
    assert not r["exact_match"] and r["lists"]["medications"]["fp"] == 1


def test_formatting_differences_are_forgiven():
    g = _gold()
    p = json.loads(target(g))
    p["sex"] = p["sex"].upper()
    p["vitals"]["temperature_f"] = g["vitals"]["temperature_f"] + 0.1
    p["vitals"]["heart_rate"] = str(g["vitals"]["heart_rate"])
    for m in p["medications"]:
        m["dose"] = m["dose"].replace(" ", "")
    assert score_doc(g, p)["exact_match"]


def test_wrong_temperature_and_invalid_json():
    g = _gold()
    p = json.loads(target(g))
    p["vitals"]["temperature_f"] = g["vitals"]["temperature_f"] + 1.0
    assert not score_doc(g, p)["fields"]["temperature_f"]
    agg = aggregate([g, g], [None, json.loads(target(g))])
    assert agg["valid_json"] == 0.5 and agg["exact_match"] == 0.5


# ---------------- costs
def test_api_cost():
    # 1000 in, 200 out on Haiku 5.5: 1000*0.10/1e6 + 200*0.50/1e6 = 0.0002
    assert costs.api_cost_per_doc("claude-haiku-5-5", 1000, 200) == pytest.approx(0.0002)
    assert costs.api_cost_per_doc("claude-sonnet-5-5", 1000, 200, batch=True) == pytest.approx(0.002)


def test_break_even_simple():
    # GPU $1/h, 10,000 docs/h; API $0.001/doc -> break-even at 24 / 0.001 = 24,000 docs/day
    assert costs.break_even_docs_per_day(0.001, 1.0, 10_000) == pytest.approx(24_000)


def test_break_even_never_when_gpu_is_dearer_per_doc():
    assert costs.break_even_docs_per_day(0.0001, 1.0, 5_000) is None


def test_break_even_needs_more_than_one_gpu():
    # one GPU carries 96,000/day at $24/day; a $3,000 training cost adds $100/day over 30 days.
    # With g GPUs the crossing is (24g + 100) / 0.0005; the first g where that fits is g = 5 -> 440,000/day
    v = costs.break_even_docs_per_day(0.0005, 1.0, 4_000, train_cost=3000)
    assert v == pytest.approx(440_000)
    assert costs.selfhost_cost_per_day(v, 1.0, 4_000, 3000) == pytest.approx(v * 0.0005)


# ---------------- training data: loss only on the answer
def test_sft_masks_prompt_and_learns_answer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(ROOT / "train"))
    from train_lora import SFTData, tiny_model_and_tokenizer, end_token
    rows = make_split(30, 1, False)
    _, tok = tiny_model_and_tokenizer(rows)
    ds = SFTData(rows[:5], tok, 4096)
    for item, r in zip(ds.items, rows):
        learned = [t for t, l in zip(item["input_ids"], item["labels"]) if l != -100]
        assert tok.decode(learned) == target(r["label"]) + end_token(tok)
        n_prompt = item["labels"].count(-100)
        assert "<think>\n\n</think>" in tok.decode(item["input_ids"][:n_prompt])


# ---------------- ICD-10-CM combination codes
def _with_dx(codes):
    g = json.loads(target(_gold()))
    g["diagnoses"] = [{"name": c, "icd10": c} for c in codes]
    return g


def test_combination_code_accepted_when_companion_present():
    g = _with_dx(["I10", "N18.31"])
    assert score_doc(g, _with_dx(["I12.9", "N18.31"]))["fields"]["diagnoses"]
    g = _with_dx(["E11.9", "G62.9"])
    assert score_doc(g, _with_dx(["E11.42"]))["fields"]["diagnoses"]


def test_combination_code_rejected_without_companion():
    g = _with_dx(["I10"])                      # no kidney disease in the note
    r = score_doc(g, _with_dx(["I12.9"]))
    assert not r["fields"]["diagnoses"] and r["lists"]["diagnoses"]["fp"] == 1


def test_invalid_codes_still_wrong():
    assert not score_doc(_with_dx(["M81.0"]), _with_dx(["M81.9"]))["fields"]["diagnoses"]
