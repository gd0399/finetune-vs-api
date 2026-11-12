"""Synthetic clinical notes with exact gold labels.

Splits
  train / val   : 3 note styles (SOAP, narrative, H&P), seen vocabulary
  test_id       : same 3 styles, new patients            -> in-distribution accuracy
  test_ood      : 2 NEW styles (telegraphic shorthand, dictation with Celsius and days),
                  Latin abbreviations, and held-out conditions/medications
                                                          -> does it generalise or memorise?

Every note also carries distractors that must NOT be extracted:
family history, ruled-out conditions and discontinued medications.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .vocab import (ALLERGIES, CONDITIONS, FREQ_ABBREV, FREQ_SPELLED, HELDOUT_CONDITIONS,
                    HELDOUT_MEDICATIONS, MALE_ONLY, MEDICATIONS, RELATIVES)

# how clinicians actually write each condition: (plain phrase, shorthand)
SPOKEN = {
    "I10": ("hypertension", "HTN"), "E11.9": ("type 2 diabetes", "T2DM"),
    "E78.5": ("hyperlipidemia", "HLD"), "J45.20": ("mild intermittent asthma", "asthma, mild intermittent"),
    "J44.9": ("COPD", "COPD"), "F32.9": ("major depressive disorder", "MDD"),
    "F41.1": ("generalized anxiety disorder", "GAD"), "E03.9": ("hypothyroidism", "hypothyroid"),
    "K21.9": ("GERD", "GERD"), "M17.9": ("osteoarthritis of the knee", "knee OA"),
    "M54.50": ("low back pain", "LBP"), "G43.909": ("migraine", "migraine"),
    "I48.91": ("atrial fibrillation", "AFib"), "I50.9": ("heart failure", "CHF"),
    "N18.31": ("chronic kidney disease stage 3a", "CKD 3a"), "E66.9": ("obesity", "obesity"),
    "J06.9": ("an upper respiratory infection", "URI"), "N39.0": ("a urinary tract infection", "UTI"),
    "J18.9": ("pneumonia", "PNA"), "J20.9": ("acute bronchitis", "acute bronchitis"),
    "E55.9": ("vitamin D deficiency", "vit D def"), "G47.00": ("insomnia", "insomnia"),
    "M81.0": ("osteoporosis", "osteoporosis"), "J30.9": ("allergic rhinitis", "allergic rhinitis"),
    "I25.10": ("coronary artery disease", "CAD"), "R73.03": ("prediabetes", "preDM"),
    "G62.9": ("peripheral neuropathy", "neuropathy"), "L20.9": ("atopic dermatitis", "atopic derm"),
    "M10.9": ("gout", "gout"), "M72.2": ("plantar fasciitis", "plantar fasciitis"),
    "G56.00": ("carpal tunnel syndrome", "CTS"), "L40.9": ("psoriasis", "psoriasis"),
    "E87.6": ("hypokalemia", "hypoK"), "J01.90": ("acute sinusitis", "acute sinusitis"),
    "D50.9": ("iron deficiency anemia", "IDA"), "N40.0": ("benign prostatic hyperplasia", "BPH"),
}


# ------------------------------------------------------------------ records
def sample_record(rng: random.Random, heldout: bool) -> dict:
    sex = rng.choice(["female", "male"])
    age = rng.randint(25, 88)
    conds = CONDITIONS + (HELDOUT_CONDITIONS if heldout else [])
    conds = [c for c in conds if c[1] not in MALE_ONLY or sex == "male"]
    meds = MEDICATIONS + (HELDOUT_MEDICATIONS if heldout else [])

    dx = rng.sample(conds, rng.randint(1, 3))
    if heldout and rng.random() < 0.6:          # make sure OOD notes really contain unseen items
        pool = [c for c in HELDOUT_CONDITIONS if c not in dx and (c[1] not in MALE_ONLY or sex == "male")]
        dx[0] = rng.choice(pool)
    rx = rng.sample(meds, rng.randint(0, 4))
    if heldout and rx and rng.random() < 0.6:
        pool = [m for m in HELDOUT_MEDICATIONS if m not in rx]
        rx[0] = rng.choice(pool)

    unused_c = [c for c in conds if c not in dx]
    unused_m = [m for m in meds if m not in rx]
    temp_f = round(rng.uniform(97.0, 100.8), 1)
    rec = {
        "age": age,
        "sex": sex,
        "chief_complaint": dx[0][2],
        "diagnoses": [{"name": n, "icd10": c} for n, c, _ in dx],
        "medications": [{"name": n, "dose": rng.choice(d), "frequency": f} for n, d, f in rx],
        "allergies": [] if rng.random() < 0.45 else rng.sample(ALLERGIES, rng.randint(1, 2)),
        "vitals": {"bp_systolic": rng.randint(102, 168), "bp_diastolic": rng.randint(62, 102),
                   "heart_rate": rng.randint(54, 112), "temperature_f": temp_f,
                   "spo2": rng.randint(91, 100)},
        "smoking_status": rng.choice(["never", "former", "current"]),
        "follow_up_weeks": None if rng.random() < 0.15 else rng.choice([1, 2, 3, 4, 6, 8, 12]),
    }
    # distractors (kept outside the label)
    meta = {
        "family_history": rng.choice(unused_c) if rng.random() < 0.5 else None,
        "relative": rng.choice(RELATIVES),
        "ruled_out": rng.choice([c for c in unused_c if c != None]) if rng.random() < 0.3 else None,
        "discontinued": rng.choice(unused_m) if unused_m and rng.random() < 0.35 else None,
        "quit_year": rng.randint(1995, 2024),
        "packs": rng.choice(["half a pack", "1 pack", "about 10 cigarettes"]),
    }
    if meta["ruled_out"] == meta["family_history"]:
        meta["ruled_out"] = None
    return {"label": rec, "meta": meta}


# ------------------------------------------------------------------ helpers
def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _spoken(code, short=False):
    return SPOKEN[code][1 if short else 0]


def _he(sex, cap=False):
    w = "she" if sex == "female" else "he"
    return w.capitalize() if cap else w


def _f2c(f):
    return round((f - 32) * 5 / 9, 1)


def _freq(rng, canon, abbrev):
    return rng.choice((FREQ_ABBREV if abbrev else FREQ_SPELLED)[canon])


def _med_line(rng, m, abbrev=False):
    name = m["name"] if abbrev else m["name"].capitalize()
    dose = m["dose"].replace(" ", "") if abbrev and rng.random() < 0.5 else m["dose"]
    return f"{name} {dose} {_freq(rng, m['frequency'], abbrev)}"


def _smoking(rng, L, meta, short=False):
    s = L["smoking_status"]
    if short:
        return {"never": rng.choice(["never smoker", "no tob", "nonsmoker"]),
                "former": rng.choice([f"ex-smoker, quit {meta['quit_year']}", "former smoker", "quit tob"]),
                "current": rng.choice([f"smokes ~{meta['packs']}/day", "current smoker", "active tob use"])}[s]
    return {"never": rng.choice(["Never smoker.", "Denies any tobacco use.", "Has never smoked."]),
            "former": rng.choice([f"Former smoker, quit in {meta['quit_year']}.",
                                  f"Quit smoking in {meta['quit_year']}."]),
            "current": rng.choice([f"Currently smokes {meta['packs']} per day.",
                                   f"Active smoker, {meta['packs']} daily."])}[s]


def _allergy(rng, L, short=False):
    a = L["allergies"]
    if short:
        return "NKDA" if not a else "All: " + ", ".join(a)
    if not a:
        return rng.choice(["No known drug allergies.", "NKDA.", "Denies allergies."])
    return rng.choice([f"Allergic to {_join(a)}.", f"Allergies: {', '.join(a)}."])


def _followup(rng, L, short=False, days=False):
    w = L["follow_up_weeks"]
    if w is None:
        return rng.choice(["f/u prn", "RTC prn"]) if short else rng.choice(
            ["Return as needed.", "Follow up as needed."])
    if days and rng.random() < 0.6:
        return f"Return to clinic in {w * 7} days."
    if short:
        return rng.choice([f"RTC {w} wks", f"f/u {w}w"])
    unit = "week" if w == 1 else "weeks"
    return rng.choice([f"Follow up in {w} {unit}.", f"Return to clinic in {w} {unit}."])


def _distractors(rng, M, sex, short=False):
    out = []
    if M["family_history"]:
        c = M["family_history"][1]
        out.append(f"FH: {M['relative']} w/ {_spoken(c, True)}" if short else
                   f"Family history is notable for {_spoken(c)} in {'her' if sex == 'female' else 'his'} "
                   f"{M['relative']}.")
    if M["discontinued"]:
        n = M["discontinued"][0]
        out.append(f"d/c'd {n} (side effects)" if short else
                   rng.choice([f"{n.capitalize()} was discontinued last month due to side effects.",
                               f"Previously took {n}, stopped because of side effects."]))
    return out


def _ruled_out(rng, M, short=False):
    if not M["ruled_out"]:
        return ""
    c = M["ruled_out"][1]
    return f" r/o {_spoken(c, True)} - neg." if short else rng.choice(
        [f" {_spoken(c).capitalize()} was considered but ruled out.", f" Workup for {_spoken(c)} was negative."])


# ------------------------------------------------------------------ note styles
def soap(rng, L, M):
    sx = L["sex"]; v = L["vitals"]
    dxs = [d["icd10"] for d in L["diagnoses"]]
    meds = "\n".join("- " + _med_line(rng, m) for m in L["medications"]) or "- None"
    plan = "\n".join(f"{i}. {_spoken(c).capitalize()}." for i, c in enumerate(dxs, 1))
    return (f"SUBJECTIVE: {L['age']}-year-old {'woman' if sx == 'female' else 'man'} presents with "
            f"{L['chief_complaint']}. {' '.join(_distractors(rng, M, sx))} {_smoking(rng, L, M)} "
            f"{_allergy(rng, L)}\n"
            f"MEDICATIONS:\n{meds}\n"
            f"OBJECTIVE: BP {v['bp_systolic']}/{v['bp_diastolic']}, HR {v['heart_rate']}, "
            f"Temp {v['temperature_f']} F, SpO2 {v['spo2']}% on room air.\n"
            f"ASSESSMENT/PLAN:\n{plan}{_ruled_out(rng, M)}\n{_followup(rng, L)}")


def narrative(rng, L, M):
    sx = L["sex"]; v = L["vitals"]; he = _he(sx, True)
    dx = _join(_spoken(d["icd10"]) for d in L["diagnoses"])
    meds = _join(_med_line(rng, m).lower() for m in L["medications"])
    med_s = f"{he} currently takes {meds}." if meds else f"{he} takes no regular medications."
    return (f"The patient is a {L['age']}-year-old {sx} who comes in today for {L['chief_complaint']}. "
            f"{he} is being managed for {dx}. {med_s} {' '.join(_distractors(rng, M, sx))} "
            f"{_smoking(rng, L, M)} {_allergy(rng, L)} On exam, blood pressure is {v['bp_systolic']}/"
            f"{v['bp_diastolic']}, pulse {v['heart_rate']}, temperature {v['temperature_f']} degrees "
            f"Fahrenheit and oxygen saturation {v['spo2']}%.{_ruled_out(rng, M)} {_followup(rng, L)}")


def hp(rng, L, M):
    sx = L["sex"]; v = L["vitals"]
    fh = M["family_history"]
    return (f"Chief Complaint: {L['chief_complaint'].capitalize()}\n"
            f"HPI: {L['age']} y/o {sx} presenting with {L['chief_complaint']}.\n"
            f"Problem List: {', '.join(_spoken(d['icd10']) for d in L['diagnoses'])}\n"
            f"Medications: {'; '.join(_med_line(rng, m) for m in L['medications']) or 'none'}"
            + (f"\nMedication changes: {_distractors(rng, {**M, 'family_history': None}, sx)[0]}"
               if M["discontinued"] else "") +
            f"\nAllergies: {_allergy(rng, L)}\n"
            f"Social History: {_smoking(rng, L, M)}\n"
            f"Family History: {(_spoken(fh[1]) + ' (' + M['relative'] + ')') if fh else 'noncontributory'}\n"
            f"Vitals: BP {v['bp_systolic']}/{v['bp_diastolic']} | HR {v['heart_rate']} | "
            f"T {v['temperature_f']} F | SpO2 {v['spo2']}%\n"
            f"Plan:{_ruled_out(rng, M)} {_followup(rng, L)}")


# --- out-of-distribution styles (never seen in training) ---
def telegraphic(rng, L, M):
    sx = L["sex"]; v = L["vitals"]
    temp = f"T {_f2c(v['temperature_f'])}C" if rng.random() < 0.5 else f"T {v['temperature_f']}"
    parts = [f"{L['age']}yo {'F' if sx == 'female' else 'M'} c/o {L['chief_complaint']}.",
             "PMH: " + ", ".join(_spoken(d["icd10"], True) for d in L["diagnoses"]) + ".",
             "Meds: " + (", ".join(_med_line(rng, m, abbrev=True) for m in L["medications"]) or "none") + "."]
    parts += [p + "." for p in _distractors(rng, M, sx, short=True)]
    parts += [_allergy(rng, L, short=True) + ".", "SH: " + _smoking(rng, L, M, short=True) + ".",
              f"VS: BP {v['bp_systolic']}/{v['bp_diastolic']} HR {v['heart_rate']} {temp} "
              f"SpO2 {v['spo2']}% RA.",
              "A/P: " + "; ".join(_spoken(d["icd10"], True) for d in L["diagnoses"]) + "."
              + _ruled_out(rng, M, short=True), _followup(rng, L, short=True) + "."]
    return " ".join(parts)


def dictation(rng, L, M):
    sx = L["sex"]; v = L["vitals"]; he = _he(sx, True)
    meds = _join(f"{m['name']} {m['dose']} {_freq(rng, m['frequency'], rng.random() < 0.5)}"
                 for m in L["medications"])
    return (f"Dictation begins. Seen today is a {L['age']} year old {'lady' if sx == 'female' else 'gentleman'} "
            f"whose main concern today is {L['chief_complaint']}, period. Active problems include "
            f"{_join(_spoken(d['icd10']) for d in L['diagnoses'])}, period. "
            + (f"Current medicines are {meds}. " if meds else "No current medicines. ")
            + " ".join(_distractors(rng, M, sx)) + " " + _smoking(rng, L, M) + " " + _allergy(rng, L) +
            f" Vital signs: pressure {v['bp_systolic']} over {v['bp_diastolic']}, heart rate "
            f"{v['heart_rate']}, temperature {_f2c(v['temperature_f'])} degrees Celsius, saturating "
            f"{v['spo2']} percent.{_ruled_out(rng, M)} {_followup(rng, L, days=True)} End of dictation.")


TRAIN_STYLES = {"soap": soap, "narrative": narrative, "hp": hp}
OOD_STYLES = {"telegraphic": telegraphic, "dictation": dictation}


def _clean(t: str) -> str:
    return "\n".join(" ".join(line.split()) for line in t.split("\n"))


def make_split(n: int, seed: int, ood: bool) -> list[dict]:
    rng = random.Random(seed)
    styles = OOD_STYLES if ood else TRAIN_STYLES
    out = []
    for i in range(n):
        r = sample_record(rng, heldout=ood)
        style = rng.choice(list(styles))
        note = _clean(styles[style](rng, r["label"], r["meta"]))
        out.append({"id": f"{'ood' if ood else 'id'}-{seed}-{i:05d}", "style": style, "note": note,
                    "label": r["label"]})
    return out


def generate(out_dir="data", n_train=2000, n_val=200, n_test=300, seed=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    splits = {"train": make_split(n_train, seed + 1, False), "val": make_split(n_val, seed + 2, False),
              "test_id": make_split(n_test, seed + 3, False), "test_ood": make_split(n_test, seed + 4, True)}
    # no identical notes may leak from train into any eval split
    train_notes = {r["note"] for r in splits["train"]}
    for k in ("val", "test_id", "test_ood"):
        splits[k] = [r for r in splits[k] if r["note"] not in train_notes]
    for k, rows in splits.items():
        with open(out / f"{k}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {k: len(v) for k, v in splits.items()}
