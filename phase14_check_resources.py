# -*- coding: utf-8 -*-
import os, glob, shutil
import pandas as pd

BASE = "/home/cihat/CEREBRAX_PROJECT"

print("=" * 90)
print(" 1. PHASE 6 COLOC 898 FALLBACK NEDENI (L245-295)")
print("=" * 90)
p6 = open(os.path.join(BASE, "phase6_reviewer2_armor_and_singlecell.py"), errors="ignore").read().splitlines()
for i in range(244, min(295, len(p6))):
    print(f"  L{i+1:03d}: {p6[i]}")

print("\n" + "=" * 90)
print(" 2. DISKTEKI REFERANS DOSYALARI (KOORDINAT, MAF, 1000G, PLINK, MAGMA, LDSC)")
print("=" * 90)
for tool in ["plink", "plink2", "magma", "Rscript"]:
    print(f"  Arac {tool:<10}: {shutil.which(tool)}")

# Sumstats ornek sutunlari
sfile = os.path.join(BASE, "data", "munged_data", "PGC_2025_MDD_Ready.sumstats.gz")
if os.path.exists(sfile):
    sdf = pd.read_csv(sfile, sep="\t", nrows=5)
    print(f"\n  Ornek sumstats sutunlari: {list(sdf.columns)}")
    print(sdf.head(3).to_string(index=False))

# Referans / harita / .bim / .frq / .bed / gtf / gene_loc dosyalari
print("\n  Diskteki referans / koordinat / LDSC matris dosyalari:")
for root, _, files in os.walk(BASE):
    for f in sorted(files):
        fl = f.lower()
        if any(k in fl for k in ["hg38", "grch38", "coord", "map", ".bim", ".frq", "ldsc_all", "rg_matrix", "gencode", "gene_len", "hm3"]):
            if not f.endswith(".py"):
                fp = os.path.join(root, f)
                print(f"    {os.path.relpath(fp, BASE)} ({os.path.getsize(fp):,} bytes)")
