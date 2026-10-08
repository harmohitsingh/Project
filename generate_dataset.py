"""
Generates a realistic Indian Home Loan dataset with real-world income ranges.
- Applicant Income:    Rs.10,000 – Rs.5,00,000 / month
- Co-applicant Income: Rs.0     – Rs.3,00,000 / month
- Loan Amount:         Rs.5L    – Rs.2Cr      (stored in thousands)
- 2,000 records with realistic approval logic
"""

import csv
import random
import math

random.seed(42)

# ── Approval logic mirrors real bank criteria ──
def decide_approval(ai, ca, la, lt, ch, edu, married, deps, self_emp, area):
    total  = ai + ca
    emi    = (la * 1000) / lt          # monthly EMI
    foir   = emi / total               # Fixed Obligation to Income Ratio
    lti    = la / (total / 12)         # Loan-to-income (months of income)

    score = 0

    # Credit history — most important
    if ch == 1.0:
        score += 40
    else:
        score -= 50   # near-instant reject without good credit

    # FOIR (EMI/Income) — banks approve if < 50%
    if foir < 0.30:   score += 20
    elif foir < 0.50: score += 10
    elif foir < 0.65: score -= 10
    else:             score -= 25

    # Loan-to-income ratio
    if lti < 24:      score += 15
    elif lti < 36:    score += 8
    elif lti < 48:    score += 0
    else:             score -= 15

    # Absolute income
    if total >= 100000: score += 15
    elif total >= 50000: score += 10
    elif total >= 25000: score += 5
    elif total < 15000:  score -= 10

    # Education
    if edu == "Graduate": score += 8

    # Employment
    if self_emp == "No":  score += 5   # salaried = stable
    else:                 score -= 3

    # Dependents
    deps_map = {"0": 0, "1": -2, "2": -5, "3+": -10}
    score += deps_map.get(deps, 0)

    # Property area
    area_map = {"Semiurban": 5, "Urban": 3, "Rural": 0}
    score += area_map.get(area, 0)

    # Married
    if married == "Yes": score += 3

    # Add some noise for realism
    score += random.gauss(0, 5)

    # Threshold: >= 40 = Approved
    return "Y" if score >= 40 else "N"


# ── Realistic Indian income distributions by segment ──
SEGMENTS = [
    # (weight, income_min, income_max, label)
    (0.15, 10_000,  25_000,  "low"),
    (0.30, 25_000,  60_000,  "lower_mid"),
    (0.30, 60_000, 150_000,  "mid"),
    (0.15,150_000, 300_000,  "upper_mid"),
    (0.10,300_000, 500_000,  "high"),
]

def sample_income():
    r = random.random()
    cum = 0
    for w, lo, hi, _ in SEGMENTS:
        cum += w
        if r <= cum:
            # log-normal within band feels realistic
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            return round(random.uniform(log_lo, log_hi))
    return random.randint(10_000, 500_000)

def exp_income(base):
    """Co-applicant income: often 0 (40%), or fraction of applicant income."""
    if random.random() < 0.40:
        return 0
    return round(base * random.uniform(0.2, 0.8) / 1000) * 1000


# ── Property / Loan mapping (realistic for India) ──
# Loan amounts in ₹ thousands
AREA_LOAN = {
    "Urban":     (3_000,  20_000),   # 30L – 2Cr
    "Semiurban": (1_500,  10_000),   # 15L – 1Cr
    "Rural":     (500,    5_000),    # 5L  – 50L
}

TERMS = [120, 180, 240, 300, 360]   # 10–30 yrs

GENDERS   = ["Male", "Female"]
MARRIED   = ["Yes", "No"]
DEPS      = ["0", "1", "2", "3+"]
EDUCATION = ["Graduate", "Not Graduate"]
SELF_EMP  = ["Yes", "No"]
AREAS     = ["Urban", "Semiurban", "Rural"]

# Realistic demographic weights
GENDER_W   = [0.62, 0.38]
MARRIED_W  = [0.65, 0.35]
DEPS_W     = [0.35, 0.30, 0.22, 0.13]
EDU_W      = [0.68, 0.32]
SEMP_W     = [0.18, 0.82]
AREA_W     = [0.35, 0.40, 0.25]


def weighted_choice(options, weights):
    r = random.random()
    cum = 0
    for o, w in zip(options, weights):
        cum += w
        if r <= cum:
            return o
    return options[-1]


records = []
for i in range(1, 2001):
    loan_id   = f"LP{i:05d}"
    gender    = weighted_choice(GENDERS,   GENDER_W)
    married   = weighted_choice(MARRIED,   MARRIED_W)
    deps      = weighted_choice(DEPS,      DEPS_W)
    edu       = weighted_choice(EDUCATION, EDU_W)
    self_emp  = weighted_choice(SELF_EMP,  SEMP_W)
    area      = weighted_choice(AREAS,     AREA_W)

    ai  = int(math.exp(sample_income()))    # applicant income (Rs/month)
    ca  = exp_income(ai)                    # co-applicant income
    lt  = random.choice(TERMS)

    # Loan amount: realistic band for area, also constrained by income
    lo_k, hi_k = AREA_LOAN[area]
    # Max loan a bank would give = ~60x monthly income (rough rule)
    max_by_income = int(ai * 60 / 1000)
    hi_k = min(hi_k, max(lo_k + 100, max_by_income))
    la = random.randint(lo_k, hi_k)         # in thousands

    ch  = 1.0 if random.random() < 0.80 else 0.0   # 80% have good credit

    status = decide_approval(ai, ca, la, lt, ch, edu, married, deps, self_emp, area)

    records.append({
        "Loan_ID":          loan_id,
        "Gender":           gender,
        "Married":          married,
        "Dependents":       deps,
        "Education":        edu,
        "Self_Employed":    self_emp,
        "ApplicantIncome":  ai,
        "CoapplicantIncome":ca,
        "LoanAmount":       la,
        "Loan_Amount_Term": lt,
        "Credit_History":   ch,
        "Property_Area":    area,
        "Loan_Status":      status,
    })

# Write CSV
out_path = "realistic_loan_dataset.csv"
fieldnames = list(records[0].keys())
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(records)

# Summary
approved = sum(1 for r in records if r["Loan_Status"] == "Y")
incomes  = [r["ApplicantIncome"] for r in records]
loans    = [r["LoanAmount"] for r in records]
print("Dataset generated:", out_path)
print("Total records    :", len(records))
print("Approved         :", approved, f"({approved/len(records)*100:.1f}%)")
print("Rejected         :", len(records)-approved, f"({(len(records)-approved)/len(records)*100:.1f}%)")
print("Income range     : Rs.", min(incomes), "-", max(incomes))
print("Loan range (K)   : Rs.", min(loans), "K -", max(loans), "K")
print("Avg income       : Rs.", round(sum(incomes)/len(incomes)))
print("Avg loan         : Rs.", round(sum(loans)/len(loans)), "K")
