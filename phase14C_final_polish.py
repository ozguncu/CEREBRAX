# -*- coding: utf-8 -*-
import os, glob, re
import pandas as pd
import numpy as np

BASE = "/home/cihat/CEREBRAX_PROJECT"
P14 = os.path.join(BASE, "phase14_bulletproof")

print("=" * 95)
print(" 1. LDSC LOGLARINDAN GERCEK BIVARIATE gcov_int (ORNEKLEM CAKISMASI) CEKIMI")
print("=" * 95)
gcov_map = {}
for lf in glob.glob(os.path.join(BASE, "**", "*.log"), recursive=True):
    try:
        txt = open(lf, errors="ignore").read()
    except Exception:
        continue
    m = re.search(r"Summary of Genetic Correlation Results\s*\n([^\n]+)\n([^\n]+)", txt)
    if not m:
        continue
    hdr = m.group(1).strip().split()
    vals = m.group(2).strip().split()
    if len(hdr) == len(vals):
        row = dict(zip(hdr, vals))
        c1 = os.path.basename(row.get("p1", "")).replace("_Ready.sumstats.gz", "").replace(".sumstats.gz", "")
        c2 = os.path.basename(row.get("p2", "")).replace("_Ready.sumstats.gz", "").replace(".sumstats.gz", "")
        try:
            gi = float(row.get("gcov_int", "nan"))
            gse = float(row.get("gcov_int_se", "nan"))
            gcov_map[tuple(sorted([c1, c2]))] = (gi, gse)
        except ValueError:
            pass

CHECK_PAIRS = [
    # Tutarlilik / Replikasyon ciftleri
    ("PGC_2025_MDD", "finngen_R11_F5_DEPRESSIO"),
    ("PGC_2026_GAD", "finngen_R11_F5_ANXIETY"),
    ("PGC_2025_BPD", "finngen_R11_F5_EMOPER"),
    ("PGC_2024_BIP", "finngen_R11_F5_BIPO"),
    ("PGC_2025_OCD", "finngen_R11_F5_OCD"),
    ("PGC_2020_PTSD", "finngen_R11_F5_PTSD"),
    ("PGC_2022_SCZ", "finngen_R11_F5_SCHZPHR"),
    ("PGC_2019_AUT", "finngen_R11_KRA_PSY_AUTISM_EXMORE"),
    ("AD_2022_Bellenguez", "AD_2021_Wightman"),
    ("ALS_2021_MinE_Full", "ALS_2021_MinE_Clin"),
    ("MS_2024_Science", "MS_2025_Shigesi"),
    # 6 MR ekseni
    ("ET_2024_Verma", "PGC_2026_GAD"),
    ("MS_2024_Science", "PGC_2024_BIP"),
    ("STROKE_2022_GIGA", "PGC_2025_MDD"),
    ("HEADACHE_Gen_Mbatchou", "PGC_2026_GAD"),
    ("MIG_2022_IHGC", "PGC_2025_MDD"),
    ("PD_2024_NatGen", "PGC_2025_MDD"),
]
ov_rows = []
for a, b in CHECK_PAIRS:
    gi, gse = gcov_map.get(tuple(sorted([a, b])), (float("nan"), float("nan")))
    ov_rows.append({"Cohort_1": a, "Cohort_2": b, "gcov_int": round(gi, 4) if pd.notna(gi) else np.nan, "gcov_int_SE": round(gse, 4) if pd.notna(gse) else np.nan})
df_ov = pd.DataFrame(ov_rows)
df_ov.to_csv(os.path.join(P14, "CEREBRAX_Phase14_SampleOverlap_gcov_int_FIXED.tsv"), sep="\t", index=False)
print(df_ov.to_string(index=False))

print("\n" + "=" * 95)
print(" 2. MR TABLOSU TAM BILIMSEL P-DEGERLERI (GAD -> HEADACHE DAHIL)")
print("=" * 95)
df_mr = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Table5_MR_10Mb_Clean.tsv"), sep="\t")
# GAD -> Headache kesin P hesabi (Beta/SE normal kuyruk)
from scipy.stats import norm
for idx, r in df_mr.iterrows():
    if r["IVW_P"] == 0.0:
        z_ivw = abs(r["IVW_Beta"] / r["IVW_SE"])
        df_mr.at[idx, "IVW_P_Exact"] = f"{2*norm.sf(z_ivw):.2e}"
    else:
        df_mr.at[idx, "IVW_P_Exact"] = f"{r['IVW_P']:.5f}"
    if r["OutlierCorr_P"] == 0.0:
        z_oc = abs(r["OutlierCorr_Beta"] / r["OutlierCorr_SE"])
        df_mr.at[idx, "OC_P_Exact"] = f"{2*norm.sf(z_oc):.2e}"
    else:
        df_mr.at[idx, "OC_P_Exact"] = f"{r['OutlierCorr_P']:.5f}"
print(df_mr[["Exposure", "Outcome", "N_IVs_10Mb", "IVW_Beta", "IVW_SE", "IVW_P_Exact", "Outliers_Rem", "OutlierCorr_Beta", "OutlierCorr_SE", "OC_P_Exact", "WM_Beta", "WM_P", "Egger_Int_P"]].to_string(index=False))

print("\n" + "=" * 95)
print(" 3. BONFERRONI & PP.H4 DUYARLILIK TABLOSU (BIRINCIL 22: 436 LOKUS / 118 CAPRAZ)")
print("=" * 95)
df_p = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")
df_c = df_p[df_p["Domain_Pair"] == "Cross_Neuro_Psych"]
for name, m_all, m_cr in [
    ("Primary (P < 5e-8; PP.H4 >= 0.75)", df_p, df_c),
    ("Strict COLOC (PP.H4 >= 0.80)", df_p[df_p["COLOC_PP_H4_Real"]>=0.80], df_c[df_c["COLOC_PP_H4_Real"]>=0.80]),
    ("Ultra-Strict COLOC (PP.H4 >= 0.90)", df_p[df_p["COLOC_PP_H4_Real"]>=0.90], df_c[df_c["COLOC_PP_H4_Real"]>=0.90]),
    ("Bonferroni-231 (P < 2.16e-10; PP.H4>=0.75)", df_p[df_p["P_PLACO_Sym"]<5e-8/231], df_c[df_c["P_PLACO_Sym"]<5e-8/231]),
    ("Bonferroni-861 (P < 5.81e-11; PP.H4>=0.75)", df_p[df_p["P_PLACO_Sym"]<5e-8/861], df_c[df_c["P_PLACO_Sym"]<5e-8/861]),
    ("Bonf-861 + PP.H4 >= 0.90", df_p[(df_p["P_PLACO_Sym"]<5e-8/861)&(df_p["COLOC_PP_H4_Real"]>=0.90)], df_c[(df_c["P_PLACO_Sym"]<5e-8/861)&(df_c["COLOC_PP_H4_Real"]>=0.90)]),
]:
    ad = (m_all["Direction_Aligned"]=="Discordant (+/-)").sum()
    cd = (m_cr["Direction_Aligned"]=="Discordant (+/-)").sum()
    print(f"  {name:<44} | All={len(m_all):3d} ({100*ad/max(1,len(m_all)):4.1f}% +/-) | Cross={len(m_cr):3d} ({100*cd/max(1,len(m_cr)):4.1f}% +/-)")

print("\n" + "=" * 95)
print(" 4. YENI VERIDE EN GUCLU CAPRAZ HASTALIK CIFT VE ALT-MAKALE GUCU KONTROLU")
print("=" * 95)
print(df_c.groupby("Pair_Name").agg(
    Loci=("Lead_SNP", "count"),
    Discordant=("Direction_Aligned", lambda s: (s=="Discordant (+/-)").sum()),
    CS95_1=("CS95_Size_Real", lambda s: (s==1).sum())
).sort_values("Loci", ascending=False).head(20).to_string())

print("\n" + "=" * 95)
print(" 5. VEP & CADD OZETI (BIRINCIL 22: 436 LOKUS)")
print("=" * 95)
print(df_p["VEP_Consequence"].value_counts().head(10).to_string())
cadd = pd.to_numeric(df_p["CADD_PHRED"], errors="coerce").dropna()
print(f"CADD gecerli={len(cadd)} | mean={cadd.mean():.2f} | median={cadd.median():.2f} | >=10: {(cadd>=10).sum()} ({100*(cadd>=10).mean():.1f}%) | >=15: {(cadd>=15).sum()} ({100*(cadd>=15).mean():.1f}%)")
