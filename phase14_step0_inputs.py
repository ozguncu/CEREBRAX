# -*- coding: utf-8 -*-
import os, glob
import pandas as pd

BASE = "/home/cihat/CEREBRAX_PROJECT"
p6 = open(os.path.join(BASE, "phase6_reviewer2_armor_and_singlecell.py"), errors="ignore").read().splitlines()
print("=== PHASE 6 GIRIS DOSYALARI (L1-80) ===")
for i in range(min(80, len(p6))):
    if any(k in p6[i] for k in [".tsv", ".csv", "read_csv", "BASE", "OUT", "DIR", "IN_"]):
        print(f"  L{i+1:03d}: {p6[i]}")

print("\n=== 2179 SATIRLI ADAY DOSYASI ARAMA ===")
for fp in sorted(glob.glob(os.path.join(BASE, "**", "*.tsv"), recursive=True)):
    try:
        n = sum(1 for _ in open(fp, errors="ignore")) - 1
        if 1790 <= n <= 2300:
            print(f"  {n:5d} satir | {os.path.relpath(fp, BASE)}")
    except Exception:
        pass

print("\n=== LDSC MATRIS DOSYALARI ===")
for fp in sorted(glob.glob(os.path.join(BASE, "ldsc_full_matrix", "*.tsv"))):
    df = pd.read_csv(fp, sep="\t", nrows=3)
    print(f"  {os.path.basename(fp)} | cols={list(df.columns)[:8]}")
