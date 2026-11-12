"""Clinical vocabulary for the synthetic notes.

Every record is generated from these tables, so the gold labels are exact.
Items marked held-out appear ONLY in the out-of-distribution test set: they test
whether a model generalises or just memorised the training vocabulary.
"""

# (name, ICD-10-CM code, typical chief complaint)
CONDITIONS = [
    ("Essential hypertension", "I10", "elevated blood pressure readings at home"),
    ("Type 2 diabetes mellitus without complications", "E11.9", "follow-up of blood sugar control"),
    ("Hyperlipidemia, unspecified", "E78.5", "review of cholesterol results"),
    ("Mild intermittent asthma, uncomplicated", "J45.20", "occasional wheezing"),
    ("Chronic obstructive pulmonary disease, unspecified", "J44.9", "shortness of breath on exertion"),
    ("Major depressive disorder, single episode, unspecified", "F32.9", "low mood and poor sleep"),
    ("Generalized anxiety disorder", "F41.1", "persistent worry and restlessness"),
    ("Hypothyroidism, unspecified", "E03.9", "fatigue and weight gain"),
    ("Gastro-esophageal reflux disease without esophagitis", "K21.9", "burning chest discomfort after meals"),
    ("Osteoarthritis of knee, unspecified", "M17.9", "knee pain"),
    ("Low back pain, unspecified", "M54.50", "lower back pain"),
    ("Migraine, unspecified, not intractable, without status migrainosus", "G43.909", "recurrent headaches"),
    ("Unspecified atrial fibrillation", "I48.91", "palpitations"),
    ("Heart failure, unspecified", "I50.9", "leg swelling and shortness of breath"),
    ("Chronic kidney disease, stage 3a", "N18.31", "follow-up of kidney function"),
    ("Obesity, unspecified", "E66.9", "weight management"),
    ("Acute upper respiratory infection, unspecified", "J06.9", "sore throat and runny nose"),
    ("Urinary tract infection, site not specified", "N39.0", "burning with urination"),
    ("Pneumonia, unspecified organism", "J18.9", "cough and fever"),
    ("Acute bronchitis, unspecified", "J20.9", "productive cough"),
    ("Vitamin D deficiency, unspecified", "E55.9", "review of lab results"),
    ("Insomnia, unspecified", "G47.00", "difficulty sleeping"),
    ("Allergic rhinitis, unspecified", "J30.9", "seasonal sneezing and congestion"),
    ("Age-related osteoporosis without current pathological fracture", "M81.0", "bone density follow-up"),
    ("Atherosclerotic heart disease of native coronary artery without angina pectoris", "I25.10",
     "cardiac follow-up"),
    ("Prediabetes", "R73.03", "review of A1c result"),
    ("Polyneuropathy, unspecified", "G62.9", "numbness in both feet"),
    ("Atopic dermatitis, unspecified", "L20.9", "itchy rash"),
]

HELDOUT_CONDITIONS = [
    ("Gout, unspecified", "M10.9", "painful swollen big toe"),
    ("Plantar fasciitis", "M72.2", "heel pain"),
    ("Carpal tunnel syndrome, unspecified upper limb", "G56.00", "numbness and tingling in the hand"),
    ("Psoriasis, unspecified", "L40.9", "scaly patches on elbows"),
    ("Hypokalemia", "E87.6", "muscle cramps"),
    ("Acute sinusitis, unspecified", "J01.90", "facial pressure and congestion"),
    ("Iron deficiency anemia, unspecified", "D50.9", "fatigue"),
    ("Benign prostatic hyperplasia without lower urinary tract symptoms", "N40.0", "prostate follow-up"),
]

MALE_ONLY = {"N40.0"}

# Canonical frequency vocabulary: the label always uses the dict KEY.
# Spelled-out phrasings appear in every note style.
FREQ_SPELLED = {
    "once daily": ["once daily", "daily", "every day", "once a day"],
    "twice daily": ["twice daily", "two times a day", "every 12 hours"],
    "three times daily": ["three times daily", "three times a day"],
    "four times daily": ["four times daily", "four times a day"],
    "at bedtime": ["at bedtime", "nightly", "every night at bedtime"],
    "every 6 hours as needed": ["every 6 hours as needed", "every six hours as needed"],
    "every 8 hours as needed": ["every 8 hours as needed", "every eight hours as needed"],
}
# Latin abbreviations appear ONLY in the out-of-distribution test styles.
FREQ_ABBREV = {
    "once daily": ["qd", "QD", "daily"],
    "twice daily": ["bid", "BID"],
    "three times daily": ["tid", "TID"],
    "four times daily": ["qid", "QID"],
    "at bedtime": ["qhs", "QHS"],
    "every 6 hours as needed": ["q6h prn", "Q6H PRN"],
    "every 8 hours as needed": ["q8h prn", "Q8H PRN"],
}

# (name, [doses], canonical frequency)
MEDICATIONS = [
    ("lisinopril", ["10 mg", "20 mg"], "once daily"),
    ("amlodipine", ["5 mg", "10 mg"], "once daily"),
    ("metformin", ["500 mg", "1000 mg"], "twice daily"),
    ("atorvastatin", ["20 mg", "40 mg"], "at bedtime"),
    ("levothyroxine", ["50 mcg", "75 mcg", "100 mcg"], "once daily"),
    ("omeprazole", ["20 mg", "40 mg"], "once daily"),
    ("sertraline", ["50 mg", "100 mg"], "once daily"),
    ("metoprolol succinate", ["25 mg", "50 mg"], "once daily"),
    ("apixaban", ["5 mg"], "twice daily"),
    ("furosemide", ["20 mg", "40 mg"], "once daily"),
    ("losartan", ["50 mg", "100 mg"], "once daily"),
    ("hydrochlorothiazide", ["12.5 mg", "25 mg"], "once daily"),
    ("gabapentin", ["100 mg", "300 mg"], "three times daily"),
    ("ibuprofen", ["400 mg", "600 mg"], "every 8 hours as needed"),
    ("acetaminophen", ["500 mg", "650 mg"], "every 6 hours as needed"),
    ("amoxicillin", ["500 mg"], "three times daily"),
    ("nitrofurantoin", ["100 mg"], "twice daily"),
    ("montelukast", ["10 mg"], "at bedtime"),
    ("escitalopram", ["10 mg", "20 mg"], "once daily"),
    ("trazodone", ["50 mg"], "at bedtime"),
    ("cholecalciferol", ["1000 IU", "2000 IU"], "once daily"),
    ("insulin glargine", ["10 units", "20 units"], "at bedtime"),
    ("tiotropium", ["18 mcg"], "once daily"),
    ("prednisone", ["20 mg", "40 mg"], "once daily"),
    ("alendronate", ["70 mg"], "once daily"),
    ("pantoprazole", ["40 mg"], "once daily"),
]

HELDOUT_MEDICATIONS = [
    ("allopurinol", ["100 mg", "300 mg"], "once daily"),
    ("tamsulosin", ["0.4 mg"], "at bedtime"),
    ("potassium chloride", ["20 mEq"], "once daily"),
    ("cephalexin", ["500 mg"], "four times daily"),
    ("duloxetine", ["30 mg", "60 mg"], "once daily"),
    ("clopidogrel", ["75 mg"], "once daily"),
]

ALLERGIES = ["penicillin", "sulfa drugs", "codeine", "latex", "shellfish", "peanuts", "aspirin",
             "iodinated contrast"]

RELATIVES = ["mother", "father", "sister", "brother", "maternal grandmother", "paternal uncle"]
