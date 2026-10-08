"""
IBM Home Loan Approval Dataset — Analysis Script
Dataset: ibm_home_loan_approval_dataset.csv
Records: 1,000 loan applications
"""

import csv
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────
# 2. Helper Utilities
# ─────────────────────────────────────────────────────────────

def safe_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback


def approval_rate(approved: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{approved / total * 100:.1f}%"


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


# ─────────────────────────────────────────────────────────────
# 3. Summary Statistics
# ─────────────────────────────────────────────────────────────

def dataset_overview(records: list[dict]) -> None:
    total = len(records)
    approved = sum(1 for r in records if r["Loan_Status"] == "Y")
    rejected = total - approved

    print_section("DATASET OVERVIEW")
    print(f"  Total Applications  : {total:,}")
    print(f"  Approved (Y)        : {approved:,}  ({approval_rate(approved, total)})")
    print(f"  Rejected (N)        : {rejected:,}  ({approval_rate(rejected, total)})")


def applicant_demographics(records: list[dict]) -> None:
    print_section("APPLICANT DEMOGRAPHICS")

    gender_counts  = Counter(r["Gender"]        for r in records)
    married_counts = Counter(r["Married"]       for r in records)
    dep_counts     = Counter(r["Dependents"]    for r in records)
    edu_counts     = Counter(r["Education"]     for r in records)
    self_emp       = Counter(r["Self_Employed"]  for r in records)

    print(f"  Gender        : Male: {gender_counts['Male']:,}  |  Female: {gender_counts['Female']:,}")
    print(f"  Marital Status: Married: {married_counts['Yes']:,}  |  Single: {married_counts['No']:,}")
    print(f"  Education     : Graduate: {edu_counts['Graduate']:,}  |  Not Graduate: {edu_counts['Not Graduate']:,}")
    print(f"  Self-Employed : Yes: {self_emp['Yes']:,}  |  No: {self_emp['No']:,}")
    print(f"  Dependents    : {dict(sorted(dep_counts.items()))}")


def financial_summary(records: list[dict]) -> None:
    print_section("FINANCIAL SUMMARY")

    applicant_incomes  = [safe_float(r["ApplicantIncome"])   for r in records]
    coapplicant_incomes = [safe_float(r["CoapplicantIncome"]) for r in records]
    loan_amounts        = [safe_float(r["LoanAmount"])         for r in records if r["LoanAmount"]]

    def avg(lst):   return sum(lst) / len(lst) if lst else 0
    def min_(lst):  return min(lst) if lst else 0
    def max_(lst):  return max(lst) if lst else 0

    print(f"  Applicant Income     : Avg: {avg(applicant_incomes):>8,.0f}  |  Min: {min_(applicant_incomes):>6,.0f}  |  Max: {max_(applicant_incomes):>6,.0f}")
    print(f"  Co-Applicant Income  : Avg: {avg(coapplicant_incomes):>8,.0f}  |  Min: {min_(coapplicant_incomes):>6,.0f}  |  Max: {max_(coapplicant_incomes):>6,.0f}")
    print(f"  Loan Amount (Rs000)   : Avg: {avg(loan_amounts):>8,.0f}  |  Min: {min_(loan_amounts):>6,.0f}  |  Max: {max_(loan_amounts):>6,.0f}")


def loan_term_distribution(records: list[dict]) -> None:
    print_section("LOAN TERM DISTRIBUTION")
    term_counts = Counter(r["Loan_Amount_Term"] for r in records if r["Loan_Amount_Term"])
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        print(f"  {term:>6} months  : {count:>4} applications  ({approval_rate(count, len(records))})")


def credit_history_analysis(records: list[dict]) -> None:
    print_section("CREDIT HISTORY vs APPROVAL")
    groups = defaultdict(list)
    for r in records:
        groups[r["Credit_History"]].append(r["Loan_Status"])

    for history, statuses in sorted(groups.items()):
        approved = statuses.count("Y")
        total    = len(statuses)
        label    = "Good (1.0)" if history == "1.0" else "Bad (0.0)" if history == "0.0" else f"Unknown ({history})"
        print(f"  {label:<15} : Approved: {approved:>4}  |  Total: {total:>4}  |  Rate: {approval_rate(approved, total)}")


def property_area_analysis(records: list[dict]) -> None:
    print_section("PROPERTY AREA vs APPROVAL")
    groups = defaultdict(list)
    for r in records:
        groups[r["Property_Area"]].append(r["Loan_Status"])

    for area, statuses in sorted(groups.items()):
        approved = statuses.count("Y")
        total    = len(statuses)
        print(f"  {area:<12} : Approved: {approved:>4}  |  Total: {total:>4}  |  Rate: {approval_rate(approved, total)}")


def education_vs_approval(records: list[dict]) -> None:
    print_section("EDUCATION vs APPROVAL")
    groups = defaultdict(list)
    for r in records:
        groups[r["Education"]].append(r["Loan_Status"])

    for edu, statuses in sorted(groups.items()):
        approved = statuses.count("Y")
        total    = len(statuses)
        print(f"  {edu:<15} : Approved: {approved:>4}  |  Total: {total:>4}  |  Rate: {approval_rate(approved, total)}")


def missing_values_report(records: list[dict]) -> None:
    print_section("MISSING VALUES REPORT")
    columns = records[0].keys() if records else []
    found_any = False
    for col in columns:
        missing = sum(1 for r in records if r[col].strip() == "")
        if missing > 0:
            print(f"  {col:<25} : {missing:>4} missing ({approval_rate(missing, len(records))})")
            found_any = True
    if not found_any:
        print("  [OK] No missing values found across all columns.")


# ─────────────────────────────────────────────────────────────
# 4. Main Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    filepath = "ibm_home_loan_approval_dataset.csv"
    print(f"\n{'#' * 55}")
    print(f"  IBM HOME LOAN APPROVAL - DATA ANALYSIS REPORT")
    print(f"{'#' * 55}")

    records = load_dataset(filepath)

    dataset_overview(records)
    applicant_demographics(records)
    financial_summary(records)
    loan_term_distribution(records)
    credit_history_analysis(records)
    property_area_analysis(records)
    education_vs_approval(records)
    missing_values_report(records)

    print(f"\n{'#' * 55}\n")


if __name__ == "__main__":
    main()
