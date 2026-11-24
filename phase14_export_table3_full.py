# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE = "/home/cihat/CEREBRAX_PROJECT"
P14 = os.path.join(BASE, "phase14_bulletproof")

df_prim = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")
df_hubs = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Table3_Primary22_HubGenes.tsv"), sep="\t")

# Standart temiz kisaltmalar (DeepReview AUT/ASD ve ham FinnGen ismi uyarisini da cozer)
SHORT = {
    "AD_2022_Bellenguez": "AD", "ALS_2021_MinE_Full": "ALS",
    "EPI_2023_ILAE_European": "EPI", "ET_2024_Verma": "ET",
    "HBS_2024_Annals": "HBS", "HEADACHE_Gen_Mbatchou": "HEADACHE",
    "MG_2024_Braun": "MG", "MIG_2022_IHGC": "MIG",
    "MS_2024_Science": "MS", "NARCO_2024_Verma": "NARCO",
    "PD_2024_NatGen": "PD", "STROKE_2022_GIGA": "STROKE",
    "PGC_2019_AUT": "ASD", "PGC_2019_TS": "TS",
    "PGC_2020_PTSD": "PTSD", "PGC_2022_SCZ": "SCZ",
    "PGC_2024_BIP": "BIP", "PGC_2025_BPD": "BPD",
    "PGC_2025_MDD": "MDD", "PGC_2025_OCD": "OCD",
    "PGC_2026_GAD": "GAD", "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE": "SUD"
}

top15_genes = df_hubs.head(15)["Gene"].tolist()
rows = []
for g in top15_genes:
    sub = df_prim[df_prim["Resolved_Gene"] == g].sort_values("P_PLACO_Sym")
    top = sub.iloc[0]
    n_tot = sub["Pair_Name"].nunique()
    n_cr = sub[sub["Domain_Pair"] == "Cross_Neuro_Psych"]["Pair_Name"].nunique()
    pos = f"chr{top['CHR']}:{int(top['BP_hg38']):,}"
    lead_snp = str(top["Lead_SNP"])
    # En guclu 3 benzersiz cift (ilk cift lider cift)
    pair_strs = []
    for _, r in sub.iterrows():
        ps = f"{SHORT.get(r['Trait_1'], r['Trait_1'])} vs. {SHORT.get(r['Trait_2'], r['Trait_2'])}"
        if ps not in pair_strs:
            pair_strs.append(ps)
    cadd = f"{float(top['CADD_PHRED']):.2f}" if pd.notna(top["CADD_PHRED"]) else "NA"
    vep = str(top["VEP_Consequence"]) if pd.notna(top["VEP_Consequence"]) else "NA"
    rows.append({
        "Gene": g,
        "GRCh38_Pos": pos,
        "Lead_SNP": lead_snp,
        "Pairs_Tot_Cross": f"{n_tot} ({n_cr})",
        "Min_P_PLACO": f"{top['P_PLACO_Sym']:.2e}",
        "Median_PP_H4": f"{sub['COLOC_PP_H4_Real'].median():.4f}",
        "CS95_CondPP": f"{int(top['CS95_Size_Real'])} ({float(top['Top_SNP_Cond_PP_H4']):.4f})",
        "VEP": vep,
        "CADD": cadd,
        "Rep_Pairs_Top3": "; ".join(pair_strs[:3])
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(os.path.join(P14, "CEREBRAX_Phase14_Table3_FULL_10Cols.tsv"), sep="\t", index=False)
print(out_df.to_string(index=False))
