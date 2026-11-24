# -*- coding: utf-8 -*-
import os
import pandas as pd

P14 = "/home/cihat/CEREBRAX_PROJECT/phase14_bulletproof"
df = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Table4_Clean_Positive_Enrichments.tsv"), sep="\t")

PATH_LIBS = ["GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human"]
CELL_LIBS = ["Allen_Brain_Atlas_10x_scRNA_2021", "PanglaoDB_Augmented_2021", "Descartes_Cell_Types_and_Tissue_2021", "GTEx_Tissues_V8_2023", "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X"]

print("=== PART A: TUM TEMIZ YOLAKLAR (7 SUTUN: Combined_Score & Genes DAHIL) ===")
p_df = df[df["Library"].isin(PATH_LIBS)].sort_values(["Adjusted_P", "P_Value"])
print(p_df[["Gene_Set", "Library", "Term", "P_Value", "Adjusted_P", "Combined_Score", "Genes"]].to_string(index=False))

print("\n=== PART B: TUM TEMIZ HUCRE / DOKU / TF SONUCLARI (7 SUTUN) ===")
c_df = df[df["Library"].isin(CELL_LIBS)].sort_values(["Adjusted_P", "P_Value"])
print(c_df[["Gene_Set", "Library", "Term", "P_Value", "Adjusted_P", "Combined_Score", "Genes"]].to_string(index=False))
