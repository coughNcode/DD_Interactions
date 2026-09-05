"""
backend/diagnose_coverage.py — DDI-Guard coverage diagnostic.
Diagnostic only — no logic changes.

Usage:
    conda run -n E --no-capture-output python3 backend/diagnose_coverage.py
"""

from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
EVIDENCE_CSV = ROOT / "ddinter_evidence.csv"
TDC_CSV      = ROOT / "data" / "processed" / "tdc_drugbank_ddi.csv"
NAME_MAP_CSV = ROOT / "data" / "drugbank_id_to_name.csv"

QUERY_DRUG = "Acetylsalicylic acid"   # Aspirin's resolved DDInter name

# ── 1. Load DDInter ───────────────────────────────────────────────────────────
print(f"Loading DDInter from {EVIDENCE_CSV}...")
df = pd.read_csv(EVIDENCE_CSV)
print(f"  {len(df):,} rows | columns: {list(df.columns)}\n")

# ── 2. Find all DDInter partners of QUERY_DRUG (bidirectional) ────────────────
a_ser = df["Drug_A"].str.strip()
b_ser = df["Drug_B"].str.strip()

mask_as_A = a_ser.str.lower() == QUERY_DRUG.lower()
mask_as_B = b_ser.str.lower() == QUERY_DRUG.lower()

partners_when_A = set(b_ser[mask_as_A].unique())
partners_when_B = set(a_ser[mask_as_B].unique())
all_partners    = sorted(partners_when_A | partners_when_B, key=str.lower)

print(f"=== DDInter partners of '{QUERY_DRUG}' ===")
print(f"  Rows where it is Drug_A : {mask_as_A.sum():,}")
print(f"  Rows where it is Drug_B : {mask_as_B.sum():,}")
print(f"  Total unique partner drugs: {len(all_partners)}\n")

for i, name in enumerate(all_partners, 1):
    # Look up severity for this pair
    rows = df[
        ((a_ser.str.lower() == QUERY_DRUG.lower()) & (b_ser == name)) |
        ((b_ser.str.lower() == QUERY_DRUG.lower()) & (a_ser == name))
    ]
    severity = rows["Level"].iloc[0] if len(rows) else "?"
    print(f"  {i:>4}. {name:<45}  [{severity}]")

# ── 3. Cross-reference with TDC target drug list ──────────────────────────────
print(f"\n=== Cross-reference: {QUERY_DRUG} partners vs TDC drug catalogue ===")

name_map = {}
if NAME_MAP_CSV.exists():
    df_nm = pd.read_csv(NAME_MAP_CSV)
    name_map = dict(zip(df_nm["drugbank_id"].astype(str), df_nm["name"].astype(str)))

tdc_names: set[str] = set(name_map.values())

partners_in_tdc     = sorted([p for p in all_partners if p in tdc_names],     key=str.lower)
partners_not_in_tdc = sorted([p for p in all_partners if p not in tdc_names], key=str.lower)

print(f"  TDC catalogue size       : {len(tdc_names):,} drugs")
print(f"  DDInter partners total   : {len(all_partners)}")
print(f"  Partners also in TDC     : {len(partners_in_tdc)}")
print(f"  Partners NOT in TDC      : {len(partners_not_in_tdc)}")

# ── 4. Check specific 'no record' complaints ──────────────────────────────────
SPOT_CHECK = ["Vinblastine", "Vindesine", "Carbamazepine", "Warfarin", "Ibuprofen",
              "Metformin", "Atorvastatin", "Clopidogrel", "Omeprazole", "Fluoxetine"]

print(f"\n=== Spot-check: is the drug in DDInter partners list? ===")
for drug in SPOT_CHECK:
    in_partners = drug in all_partners
    in_tdc      = drug in tdc_names
    # Also check case-insensitively in DDInter vocabulary directly
    all_ddinter_names_lower = {
        n.lower().strip()
        for col in ("Drug_A", "Drug_B")
        for n in df[col].dropna().unique()
    }
    in_ddinter_vocab = drug.lower().strip() in all_ddinter_names_lower
    status = "✅ FOUND pair" if in_partners else ("❌ not a partner (genuine no_record)" if in_ddinter_vocab else "⚠️  not in DDInter vocab at all")
    print(f"  {drug:<30}  TDC={in_tdc!s:<5}  DDInterVocab={in_ddinter_vocab!s:<5}  {status}")

# ── 5. TDC drugs NOT covered in DDInter at all ───────────────────────────────
print(f"\n=== TDC drugs with NO DDInter entry at all (not a partner, not in vocab) ===")
all_ddinter_names = {
    n.strip()
    for col in ("Drug_A", "Drug_B")
    for n in df[col].dropna().unique()
}
all_ddinter_names_lower = {n.lower() for n in all_ddinter_names}

not_in_ddinter = sorted(
    [name for name in tdc_names if name.lower() not in all_ddinter_names_lower],
    key=str.lower
)
print(f"  TDC drugs not in DDInter at all: {len(not_in_ddinter)} / {len(tdc_names)}")
print(f"  First 30:")
for name in not_in_ddinter[:30]:
    print(f"    - {name}")
if len(not_in_ddinter) > 30:
    print(f"  ... and {len(not_in_ddinter)-30} more.")
