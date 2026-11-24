# -*- coding: utf-8 -*-
import os, glob, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
import scipy, matplotlib

P14 = "/home/cihat/CEREBRAX_PROJECT/phase14_bulletproof"
OUT = os.path.join(P14, "MASTER_NUMBERS_FOR_MANUSCRIPT.txt")
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

log("=== 0. YAZILIM SURUMLERI & KLASOR DOSYALARI ===")
log(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | SciPy: {scipy.__version__} | Pandas: {pd.__version__} | Matplotlib: {matplotlib.__version__}")
files = sorted(os.listdir(P14))
log("Dosyalar: " + ", ".join(files))

# 1. PRIMARY 22 DOSYASI
df_prim = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")
log("\n=== 1. SUBTHRESHOLD & CS95 & 17q21.31 KONTROLU (PRIMARY 22) ===")
z_gws = norm.isf(2.5e-8)  # 5.45131
z1 = pd.to_numeric(df_prim["Z_Trait1"], errors="coerce").abs()
z2 = pd.to_numeric(df_prim["Z_Trait2_Aligned"], errors="coerce").abs()
sub_both = (z1 < z_gws) & (z2 < z_gws)
sub_either = (z1 < z_gws) | (z2 < z_gws)
log(f"Toplam Primary 22 lokus: {len(df_prim)}")
log(f"Subthreshold (IKISINDE DE P >= 5e-8): {int(sub_both.sum())} / {len(df_prim)} ({100*sub_both.mean():.1f}%)")
log(f"Subthreshold (EN AZ BIRINDE P >= 5e-8): {int(sub_either.sum())} / {len(df_prim)} ({100*sub_either.mean():.1f}%)")
log(f"Medyan CS95: {df_prim['CS95_Size_Real'].median():.1f} | IQR: {df_prim['CS95_Size_Real'].quantile(0.25):.1f}-{df_prim['CS95_Size_Real'].quantile(0.75):.1f} | CS95==1: {int((df_prim['CS95_Size_Real']==1).sum())} / {len(df_prim)} ({100*(df_prim['CS95_Size_Real']==1).mean():.1f}%)")

# 17q21.31 ayni ciftte 2 lokus var mi kontrolu
chr17_inv = df_prim[(df_prim["CHR"].astype(str).str.replace("chr","") == "17") & (df_prim["BP_hg38"].between(43500000, 47000000))]
pair_key = df_prim[["Trait_1", "Trait_2"]].apply(lambda r: tuple(sorted([r.iloc[0], r.iloc[1]])), axis=1)
dup17 = chr17_inv.assign(PKey=pair_key.loc[chr17_inv.index]).groupby("PKey").size()
log(f"chr17:43.5-47.0 Mb ayni ciftte >1 lokus sayisi: {int((dup17 > 1).sum())} (Toplam 17q21.31 lokusu: {len(chr17_inv)})")

# 2. KOHORT BAZLI CROSS-DOMAIN / TOTAL PRIMARY 22
log("\n=== 2. PRIMARY 22 KOHORT BAZLI CROSS-DOMAIN / TOPLAM LOKUS ===")
all_traits = sorted(set(df_prim["Trait_1"]).union(set(df_prim["Trait_2"])))
log("Primary 22 Trait isimleri: " + ", ".join(all_traits))

def find_trait(substr):
    return [t for t in all_traits if substr.lower() in t.lower()]

targets = {
    "GAD": find_trait("GAD"),
    "ET": find_trait("ET_2024"),
    "OCD": find_trait("OCD"),
    "PTSD": find_trait("PTSD"),
    "BPD": find_trait("BPD"),
    "SUD": find_trait("SUBSTANCE"),
    "HEADACHE_OR_MIG": find_trait("HEADACHE") + find_trait("MIG"),
    "STROKE": find_trait("STROKE"),
}
for lbl, tlist in targets.items():
    sub = df_prim[df_prim["Trait_1"].isin(tlist) | df_prim["Trait_2"].isin(tlist)]
    cr = sub[sub["Domain_Pair"] == "Cross_Neuro_Psych"]
    log(f"  {lbl:<16} ({','.join(tlist)}): {len(cr)} cross-domain / {len(sub)} total primary")

# 3. TABLO 2: 5 PSIKIYATRIK FAKTOR x 4 NOROLOJIK SINIF (PRIMARY 22)
log("\n=== 3. TABLO 2 (5 FAKTOR x 4 NOROLOJIK SINIF - PRIMARY 22) ===")
factors = {
    "F1_Compulsive": find_trait("AN_") + find_trait("OCD") + find_trait("TS"),
    "F2_Schiz_Bip": find_trait("SCZ") + find_trait("BIP"),
    "F3_Neurodev": find_trait("ADHD") + find_trait("AUT") + find_trait("ASD"),
    "F4_Internalizing": find_trait("MDD") + find_trait("GAD") + find_trait("PTSD") + find_trait("BPD"),
    "F5_Substance": find_trait("SUBSTANCE"),
}
neuro_grp = {
    "Neurodegenerative": find_trait("AD_") + find_trait("ALS") + find_trait("LBD"),
    "Movement": find_trait("PD_") + find_trait("ET_2024"),
    "Neurovasc_Headache": find_trait("STROKE") + find_trait("MIG") + find_trait("HEADACHE") + find_trait("HBS"),
    "Neuroimmune_Excit": find_trait("MS_") + find_trait("EPI") + find_trait("MG_"),
}
log("Faktor eslesmeleri: " + str({k: sorted(set(v)) for k,v in factors.items()}))
log("Noroloji eslesmeleri: " + str({k: sorted(set(v)) for k,v in neuro_grp.items()}))

df_cr = df_prim[df_prim["Domain_Pair"] == "Cross_Neuro_Psych"]
for fk, fset in factors.items():
    fset = set(fset)
    row_str = []
    tot_f = 0
    for nk, nset in neuro_grp.items():
        nset = set(nset)
        m = df_cr[((df_cr["Trait_1"].isin(fset)) & (df_cr["Trait_2"].isin(nset))) |
                  ((df_cr["Trait_2"].isin(fset)) & (df_cr["Trait_1"].isin(nset)))]
        agon = int((m["Direction_Raw"].astype(str).str.contains(r"\+/\+|Concord|Agon", case=False, regex=True)).sum()) if "Direction_Raw" in m.columns else -1
        if agon == -1:
            dcol = [c for c in m.columns if "Dir" in c or "Conc" in c or "Effect_Dir" in c][0]
            agon = int((m[dcol].astype(str).str.contains(r"\+/\+|Concord|Agon", case=False, regex=True)).sum())
        row_str.append(f"{nk}={len(m)} (+/+:{agon}, +/-:{len(m)-agon})")
        tot_f += len(m)
    log(f"  {fk:<18} (Top={tot_f}): " + " | ".join(row_str))

f4_all = df_cr[df_cr["Trait_1"].isin(set(factors["F4_Internalizing"])) | df_cr["Trait_2"].isin(set(factors["F4_Internalizing"]))]
gcol = "Resolved_Gene" if "Resolved_Gene" in df_cr.columns else [c for c in df_cr.columns if "Gene" in c][0]
log("F4 Cross-Domain Top 10 Gen:\n" + f4_all[gcol].value_counts().head(10).to_string())

# 4. ELEME HUNISI (ALL CANDIDATES DOSYASINI OTOMATIK BUL)
log("\n=== 4. ELEME HUNISI (ADAY -> PLACO -> COLOC) ===")
cand_files = [f for f in files if f.endswith(".tsv") and ("all" in f.lower() or "cand" in f.lower() or "861" in f.lower() or "master" in f.lower())]
for cf in cand_files:
    try:
        tmp = pd.read_csv(os.path.join(P14, cf), sep="\t")
        if "P_PLACO_Sym" in tmp.columns and "COLOC_PP_H4_Real" in tmp.columns:
            log(f"Dosya: {cf} | Satir={len(tmp)}")
            if "Pair_Tier" in tmp.columns:
                for tier, g in tmp.groupby("Pair_Tier"):
                    f_pl = int((g["P_PLACO_Sym"] >= 5e-8).sum())
                    p_pl = g[g["P_PLACO_Sym"] < 5e-8]
                    f_co = int((p_pl["COLOC_PP_H4_Real"] < 0.75).sum())
                    keep = int((p_pl["COLOC_PP_H4_Real"] >= 0.75).sum())
                    log(f"  Tier={tier}: Aday={len(g)} | PLACO>=5e-8 elenen={f_pl} | COLOC<0.75 elenen={f_co} | Kalan={keep}")
            else:
                f_pl = int((tmp["P_PLACO_Sym"] >= 5e-8).sum())
                p_pl = tmp[tmp["P_PLACO_Sym"] < 5e-8]
                f_co = int((p_pl["COLOC_PP_H4_Real"] < 0.75).sum())
                keep = int((p_pl["COLOC_PP_H4_Real"] >= 0.75).sum())
                log(f"  Genel: Aday={len(tmp)} | PLACO>=5e-8 elenen={f_pl} | COLOC<0.75 elenen={f_co} | Kalan={keep}")
    except Exception as e:
        log(f"  ({cf} atlandi: {e})")

# 5. TAM TABLO 5 (MR - BILIMSEL GOSTERIM + GCOV_INT + EKSIK YON)
log("\n=== 5. TAM TABLO 5 (MR - TUM SUTUNLAR & BILIMSEL GOSTERIM) ===")
mr_files = [f for f in files if "mr" in f.lower() and f.endswith(".tsv")]
for mf in mr_files:
    try:
        mr = pd.read_csv(os.path.join(P14, mf), sep="\t")
        log(f"MR Dosyasi: {mf} ({len(mr)} satir)")
        log("MR Kolonlar: " + ", ".join(mr.columns))
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.width", 260)
        for _, r in mr.iterrows():
            vals = []
            for c in mr.columns:
                v = r[c]
                if isinstance(v, float) and ("P" in c or "p" in c) and not pd.isna(v):
                    vals.append(f"{c}={v:.2e}")
                elif isinstance(v, float) and not pd.isna(v):
                    vals.append(f"{c}={v:.4f}")
                else:
                    vals.append(f"{c}={v}")
            log("  " + " | ".join(vals))
    except Exception as e:
        log(f"  MR okuma hatasi ({mf}): {e}")

# 6. LDSC WITHIN-DISORDER 4 BOS SATIR + PURE-EUROPEAN 171-PAIR KONTROLU
log("\n=== 6. LDSC WITHIN-DISORDER 4 EKSIK CIFT & PURE-EUR 171-PAIR KONTROLU ===")
ldsc_cands = glob.glob("/home/cihat/CEREBRAX_PROJECT/**/*ldsc*rg*.tsv", recursive=True) + glob.glob("/home/cihat/CEREBRAX_PROJECT/**/*LDSC*.tsv", recursive=True)
ldsc_cands = sorted(set(ldsc_cands))
log("Bulunan LDSC dosyalari: " + ", ".join(ldsc_cands[:10]))
for lf in ldsc_cands:
    try:
        ld = pd.read_csv(lf, sep="\t")
        cols = set(ld.columns)
        if ("rg" in cols or "LDSC_rg" in cols) and len(ld) >= 150:
            t1c = [c for c in ld.columns if c in ("Trait_1","p1","Trait1")][0]
            t2c = [c for c in ld.columns if c in ("Trait_2","p2","Trait2")][0]
            rgc = "rg" if "rg" in cols else "LDSC_rg"
            sec = "se" if "se" in cols else ([c for c in ld.columns if "se" in c.lower()][0] if any("se" in c.lower() for c in ld.columns) else None)
            pc = "p" if "p" in cols else ([c for c in ld.columns if c.lower() in ("p","p_val","ldsc_p","pval")][0] if any(c.lower() in ("p","p_val","ldsc_p","pval") for c in ld.columns) else None)
            log(f"LDSC Dosya: {lf} ({len(ld)} satir, rg_col={rgc})")
            check_pairs = [
                ("PGC_2025_MDD", "finngen_R11_F5_DEPRESSIO"),
                ("PGC_2024_BIP", "finngen_R11_F5_BIPO"),
                ("PGC_2022_SCZ", "finngen_R11_F5_SCHZPHR"),
                ("finngen_R11_F5_SUBSNOALCO", "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE"),
            ]
            for a, b in check_pairs:
                m = ld[((ld[t1c].astype(str).str.contains(a)) & (ld[t2c].astype(str).str.contains(b))) |
                       ((ld[t1c].astype(str).str.contains(b)) & (ld[t2c].astype(str).str.contains(a)))]
                if len(m) > 0:
                    r0 = m.iloc[0]
                    log(f"  BULUNDU {a} vs {b}: rg={r0[rgc]} | se={r0[sec] if sec else 'NA'} | p={r0[pc] if pc else 'NA'}")
                else:
                    log(f"  YOK     {a} vs {b}")
            break
    except Exception as e:
        log(f"  LDSC hata ({lf}): {e}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log(f"\n>>> Tum ozet kaydedildi: {OUT}")
