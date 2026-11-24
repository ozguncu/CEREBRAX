# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest

BASE = "/home/cihat/CEREBRAX_PROJECT"
P14 = os.path.join(BASE, "phase14_bulletproof")

df_val = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Validated_All861.tsv"), sep="\t")
df_prim = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")
df_rep = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_CrossBiobank_Replication_Tests.tsv"), sep="\t")
df_ldsc = pd.read_csv(os.path.join(BASE, "ldsc_full_matrix", "CEREBRAX_All_861_Pairs_Results.tsv"), sep="\t")

ldsc_map = {}
for _, r in df_ldsc.iterrows():
    key = tuple(sorted([str(r["Trait_1"]), str(r["Trait_2"])]))
    ldsc_map[key] = (r["rg"], r["se"], r["p"])

print("=" * 95)
print(" PART A. WITHIN-DISORDER CALIBRATION (HASTALIK BAZLI DETAY + LDSC rg)")
print("=" * 95)
df_w = df_val[df_val["Pair_Tier"] == "Within_Disorder_Calibration"].copy()
part_a_rows = []
for pname, grp in df_w.groupby("Pair_Name"):
    t1, t2 = grp["Trait_1"].iloc[0], grp["Trait_2"].iloc[0]
    rg, se, p = ldsc_map.get(tuple(sorted([t1, t2])), (np.nan, np.nan, np.nan))
    n_tot = len(grp)
    n_conc = int((grp["Direction_Aligned"] == "Concordant (+/+)").sum())
    part_a_rows.append({
        "Matched_Cohort_Pair": f"{t1} vs. {t2}",
        "LDSC_rg": f"{rg:.4f}" if pd.notna(rg) else "—",
        "LDSC_SE": f"{se:.4f}" if pd.notna(se) else "—",
        "LDSC_P": f"{p:.2e}" if pd.notna(p) else "—",
        "Colocalized_Loci": n_tot,
        "Concordance": f"{100*n_conc/n_tot:.1f}% ({n_conc}/{n_tot})"
    })
df_pa = pd.DataFrame(part_a_rows).sort_values("Colocalized_Loci", ascending=False)
print(df_pa.to_string(index=False))
print(f"Toplam Within-Disorder cift: {len(df_pa)} | Toplam lokus: {df_pa['Colocalized_Loci'].sum()}")

print("\n" + "=" * 95)
print(" PART B. CROSS-BIOBANK CONSISTENCY (GLOBAL LDSC rg + VARIANT + DEDUPLICATED SNP)")
print("=" * 95)
REP_MAP = {
    "PGC_2025_MDD": "finngen_R11_F5_DEPRESSIO",
    "PGC_2026_GAD": "finngen_R11_F5_ANXIETY",
    "PGC_2025_BPD": "finngen_R11_F5_EMOPER",
    "PGC_2024_BIP": "finngen_R11_F5_BIPO",
    "PGC_2025_OCD": "finngen_R11_F5_OCD",
    "PGC_2020_PTSD": "finngen_R11_F5_PTSD",
    "PGC_2022_SCZ": "finngen_R11_F5_SCHZPHR",
    "PGC_2019_AUT": "finngen_R11_KRA_PSY_AUTISM_EXMORE",
    "AD_2022_Bellenguez": "AD_2021_Wightman",
    "ALS_2021_MinE_Full": "ALS_2021_MinE_Clin",
    "MS_2024_Science": "MS_2025_Shigesi",
}
df_p_ldsc = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_LDSC_231Pairs.tsv"), sep="\t")
rg_disc, rg_bench = [], []
for _, r in df_p_ldsc.iterrows():
    t1, t2, rg1 = r["Trait_1"], r["Trait_2"], r["rg"]
    if pd.isna(rg1) or abs(rg1) > 1.5:
        continue
    # Bir ucu Primary diger ucu karsilik gelen Benchmark olan capraz ciftleri karsilastir
    for a, b in [(t1, t2), (t2, t1)]:
        if a in REP_MAP:
            ra = REP_MAP[a]
            if ra != b:
                rg2, se2, _ = ldsc_map.get(tuple(sorted([ra, b])), (np.nan, np.nan, np.nan))
                if pd.notna(rg2) and pd.notna(se2) and abs(rg2) <= 1.5 and se2 < 0.5:
                    rg_disc.append(float(rg1))
                    rg_bench.append(float(rg2))

rg_disc, rg_bench = np.array(rg_disc), np.array(rg_bench)
r_rg, p_rg = pearsonr(rg_disc, rg_bench)
same_rg = int(np.sum((rg_disc * rg_bench) > 0))
bp_rg = binomtest(same_rg, len(rg_disc), 0.5).pvalue
print(f"  Global LDSC rg Consistency (n={len(rg_disc)} pairs): r = {r_rg:.4f} (P = {p_rg:.2e}) | Dir Conc = {same_rg}/{len(rg_disc)} ({100*same_rg/len(rg_disc):.1f}%, P = {bp_rg:.2e})")

df_rep_uniq = df_rep.sort_values("Z_Disc", key=lambda s: s.abs(), ascending=False).drop_duplicates(subset=["Lead_SNP"], keep="first")
nom_u = int(df_rep_uniq["Nom_Sig"].sum())
print(f"  Deduplicated Unique Lead SNPs (n={len(df_rep_uniq)}): Nominal P<0.05 = {nom_u}/{len(df_rep_uniq)} ({100*nom_u/len(df_rep_uniq):.1f}%)")

print("\n" + "=" * 95)
print(" PART C. SENSITIVITY ANALYSIS (UNIQUE GENES & EXACT +/+ vs +/- COUNTS)")
print("=" * 95)
df_c = df_prim[df_prim["Domain_Pair"] == "Cross_Neuro_Psych"]
thresholds = [
    ("Primary (P < 5e-8; PP.H4 >= 0.75)", df_prim, df_c),
    ("Strict COLOC (P < 5e-8; PP.H4 >= 0.80)", df_prim[df_prim["COLOC_PP_H4_Real"]>=0.80], df_c[df_c["COLOC_PP_H4_Real"]>=0.80]),
    ("Ultra-Strict COLOC (P < 5e-8; PP.H4 >= 0.90)", df_prim[df_prim["COLOC_PP_H4_Real"]>=0.90], df_c[df_c["COLOC_PP_H4_Real"]>=0.90]),
    ("Bonferroni-231 (P < 2.16e-10; PP.H4 >= 0.75)", df_prim[df_prim["P_PLACO_Sym"]<5e-8/231], df_c[df_c["P_PLACO_Sym"]<5e-8/231]),
    ("Bonferroni-861 (P < 5.81e-11; PP.H4 >= 0.75)", df_prim[df_prim["P_PLACO_Sym"]<5e-8/861], df_c[df_c["P_PLACO_Sym"]<5e-8/861]),
    ("Bonf-861 + Ultra-Strict (PP.H4 >= 0.90)", df_prim[(df_prim["P_PLACO_Sym"]<5e-8/861)&(df_prim["COLOC_PP_H4_Real"]>=0.90)], df_c[(df_c["P_PLACO_Sym"]<5e-8/861)&(df_c["COLOC_PP_H4_Real"]>=0.90)]),
]
for lbl, sa, sc in thresholds:
    g_all = sa["Resolved_Gene"].dropna().astype(str)
    g_hgnc = g_all[~g_all.str.startswith(("ENSG", "LOC_chr"))]
    ac = int((sa["Direction_Aligned"]=="Concordant (+/+)").sum())
    ad = int((sa["Direction_Aligned"]=="Discordant (+/-)").sum())
    cc = int((sc["Direction_Aligned"]=="Concordant (+/+)").sum())
    cd = int((sc["Direction_Aligned"]=="Discordant (+/-)").sum())
    print(f"  {lbl:<45} | Loci={len(sa):3d} ({100*len(sa)/436:5.1f}%) | Genes={g_all.nunique():3d} ({g_hgnc.nunique():3d} HGNC) | All: {ac} vs {ad} ({100*ad/max(1,len(sa)):4.1f}%) | Cross={len(sc):3d} ({100*len(sc)/118:5.1f}%): {cc} vs {cd} ({100*cd/max(1,len(sc)):4.1f}%)")
