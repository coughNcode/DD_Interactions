"""
Batch-resolve 1,705 DrugBank IDs → common names via PubChem synonyms API.
Uses ThreadPoolExecutor for parallel requests (~60s total).
Output: data/drugbank_id_to_name.csv  (columns: drugbank_id, name)
"""

import pandas as pd
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT    = Path(__file__).resolve().parent.parent
TDC_CSV = ROOT / "data" / "processed" / "tdc_drugbank_ddi.csv"
OUT_CSV = ROOT / "data" / "drugbank_id_to_name.csv"

MAX_WORKERS = 20   # stay well under PubChem's 5 req/s per-IP; 20 parallel is fine
TIMEOUT     = 15


def fetch_name(db_id: str) -> tuple[str, str | None]:
    """Return (db_id, preferred_name_or_None)."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        f"/compound/xref/RegistryID/{db_id}/synonyms/JSON"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            info = r.json().get("InformationList", {}).get("Information", [])
            if info:
                syns = info[0].get("Synonym", [])
                if syns:
                    # First synonym is the canonical/preferred common name
                    return db_id, syns[0]
        return db_id, None
    except Exception:
        return db_id, None


def main():
    df = pd.read_csv(TDC_CSV)
    all_ids = sorted(set(df["Drug1_ID"].astype(str)) | set(df["Drug2_ID"].astype(str)))
    print(f"Resolving {len(all_ids):,} DrugBank IDs via PubChem ({MAX_WORKERS} parallel workers)...")

    id_to_name: dict = {}
    failed: list = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_name, db_id): db_id for db_id in all_ids}
        for future in as_completed(futures):
            db_id, name = future.result()
            done_count += 1
            if name:
                id_to_name[db_id] = name
            else:
                failed.append(db_id)
            if done_count % 100 == 0 or done_count == len(all_ids):
                print(f"  {done_count}/{len(all_ids)} done | resolved: {len(id_to_name):,}", flush=True)

    # ── Report ────────────────────────────────────────────────────────────────
    total = len(all_ids)
    resolved = len(id_to_name)
    print(f"\nResolved: {resolved:,} / {total:,} ({resolved/total*100:.1f}%)")
    if failed:
        print(f"Failed ({len(failed)}): {sorted(failed)}")

    out = pd.DataFrame(
        sorted(id_to_name.items()), columns=["drugbank_id", "name"]
    )
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")

    # ── Overlap with DDInter ──────────────────────────────────────────────────
    df_ev = pd.read_csv(ROOT / "data" / "processed" / "ddinter_evidence.csv")
    ddi_names = set(df_ev["Drug_A"].astype(str).str.lower().str.strip()) | \
                set(df_ev["Drug_B"].astype(str).str.lower().str.strip())
    tdc_names_lower = set(n.lower().strip() for n in id_to_name.values())
    overlap = tdc_names_lower & ddi_names
    print(f"\nName-overlap (resolved TDC names ↔ DDInter names):")
    print(f"  TDC resolved names: {len(tdc_names_lower):,}")
    print(f"  DDInter names:       {len(ddi_names):,}")
    print(f"  Overlap:             {len(overlap):,} ({len(overlap)/len(tdc_names_lower)*100:.1f}% of TDC)")
    print(f"\nSample overlapping names: {sorted(overlap)[:10]}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
