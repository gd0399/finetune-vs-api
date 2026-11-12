"""The one extraction prompt every model gets, so the comparison is fair."""
import json
import re

SYSTEM = """You extract structured data from a clinical note. Reply with ONE JSON object and nothing else.

Schema:
{
  "age": integer,
  "sex": "female" | "male",
  "chief_complaint": string (the reason for today's visit, in plain words, lowercase),
  "diagnoses": [{"name": string (official ICD-10-CM title), "icd10": string (ICD-10-CM code, e.g. "E11.9")}],
  "medications": [{"name": string (generic name, lowercase), "dose": string (e.g. "10 mg"),
                   "frequency": one of "once daily" | "twice daily" | "three times daily" | "four times daily" |
                                "at bedtime" | "every 6 hours as needed" | "every 8 hours as needed"}],
  "allergies": [string] (lowercase; [] if none),
  "vitals": {"bp_systolic": integer, "bp_diastolic": integer, "heart_rate": integer,
             "temperature_f": number (degrees Fahrenheit, 1 decimal; convert from Celsius if needed),
             "spo2": integer},
  "smoking_status": "never" | "former" | "current",
  "follow_up_weeks": integer or null (convert days to weeks; null if "as needed")
}

Rules:
- diagnoses = the patient's current conditions only. Do NOT include family history or conditions
  that were ruled out.
- medications = current medications only. Do NOT include discontinued or stopped medications.
- Use the most specific correct ICD-10-CM code for the condition as written."""


def user_message(note: str) -> str:
    return f"Clinical note:\n\n{note}"


def messages(note: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(note)}]


def target(label: dict) -> str:
    """What the fine-tuned model learns to output: compact JSON."""
    return json.dumps(label, separators=(",", ":"))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse(text: str):
    """Return the first JSON object in a reply, or None if there isn't a valid one."""
    if text is None:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
