# -*- coding: utf-8 -*-
import os
import pandas as pd

P14 = "/home/cihat/CEREBRAX_PROJECT/phase14_bulletproof"
df_all = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_All_Candidates_Evaluated.tsv"), sep="\t")
df_prim = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")

p_cand = df_all[df_all["Pair_Tier"] == "Primary_22"]
fail_placo = int((p_cand["P_PLACO_Sym"] >= 5e-8).sum())
pass_placo = p_cand[p_cand["P_PLACO_Sym"] < 5e-8]
fail_coloc = int((pass_placo["COLOC_PP_H4_Real"] < 0.75).sum())
print(f"Primary 22 Eleme: Aday={len(p_cand)} | PLACO>=5e-8 elenen={fail_placo} | COLOC<0.75 elenen={fail_coloc} | Kalan={len(df_prim)}")

targets = {
    "GAD": "PGC_2026_GAD",
    "ET": "ET_2024_Verma",
    "OCD": "PGC_2025_OCD",
    "PTSD": "PGC_2020_PTSD",
    "BPD": "PGC_2025_BPD",
    "SUD": "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE",
}
print("\n=== 6 YENI KOHORT: CROSS-DOMAIN / TOPLAM PRIMARY 22 ===")
for lbl, code in targets.items():
    sub = df_prim[(df_prim["Trait_1"] == code) | (df_prim["Trait_2"] == code)]
    cr = sub[sub["Domain_Pair"] == "Cross_Neuro_Psych"]
    print(f"  {lbl:<5}: {len(cr)} cross-domain / {len(sub)} total primary loci")

F4 = {"PGC_2025_MDD", "PGC_2026_GAD", "PGC_2020_PTSD", "PGC_2025_BPD"}
NEUROVASC = {"STROKE_2022_GIGA", "MIG_2022_IHGC", "HEADACHE_Gen_Mbatchou", "HBS_2024_Annals"}
MOVEMENT = {"PD_2024_NatGen", "ET_2024_Verma"}

df_cr = df_prim[df_prim["Domain_Pair"] == "Cross_Neuro_Psych"]
f4_nv = df_cr[((df_cr["Trait_1"].isin(F4)) & (df_cr["Trait_2"].isin(NEUROVASC))) | ((df_cr["Trait_2"].isin(F4)) & (df_cr["Trait_1"].isin(NEUROVASC)))]
f4_mv = df_cr[((df_cr["Trait_1"].isin(F4)) & (df_cr["Trait_2"].isin(MOVEMENT))) | ((df_cr["Trait_2"].isin(F4)) & (df_cr["Trait_1"].isin(MOVEMENT)))]
f4_all = df_cr[(df_cr["Trait_1"].isin(F4)) | (df_cr["Trait_2"].isin(F4))]
print(f"\nF4 (Internalizing) Cross-Domain Toplam: {len(f4_all)} | Neurovasc/Headache: {len(f4_nv)} | Movement: {len(f4_mv)}")
print("F4 Cross-Domain Top 10 Hub Gen:")
print(f4_all["Resolved_Gene"].value_counts().head(10).to_string())
