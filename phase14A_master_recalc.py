# -*- coding: utf-8 -*-
import os, gzip, math, bisect, glob, time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import kn, logsumexp
from scipy.integrate import quad
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
BASE = "/home/cihat/CEREBRAX_PROJECT"
OUT_DIR = os.path.join(BASE, "phase14_bulletproof")
os.makedirs(OUT_DIR, exist_ok=True)

P6_CAND = os.path.join(BASE, "phase6_armor_and_singlecell", "CEREBRAX_Master_Consolidated_Decorrelated_RegionalCOLOC.tsv")
P11_FILE = os.path.join(BASE, "phase11_nature_neuro_final_arsenal", "CEREBRAX_1800_Loci_ALLELE_ALIGNED_FINAL.tsv")
LDSC_FILE = os.path.join(BASE, "ldsc_full_matrix", "CEREBRAX_All_861_Pairs_Results.tsv")
QC_FILE = os.path.join(BASE, "phase11_nature_neuro_final_arsenal", "CEREBRAX_Table1_Final_Real_LDSC_h2.tsv")
COORD_FILE = os.path.join(BASE, "ref_data", "hm3_hg38_chr_bp_to_rsid.tsv.gz")
SUMSTATS_DIR = os.path.join(BASE, "data", "munged_data")

# =============================================================================
# 1. IKI KATMANLI MIMARI: 22 TEKRARSIZ BIRINCIL + 20 TUTARLILIK/KALIBRASYON
# =============================================================================
PRIMARY_NEURO = {
    "AD_2022_Bellenguez", "ALS_2021_MinE_Full", "EPI_2023_ILAE_European",
    "ET_2024_Verma", "HBS_2024_Annals", "HEADACHE_Gen_Mbatchou",
    "MG_2024_Braun", "MIG_2022_IHGC", "MS_2024_Science",
    "NARCO_2024_Verma", "PD_2024_NatGen", "STROKE_2022_GIGA"
}
PRIMARY_PSYCH = {
    "PGC_2019_AUT", "PGC_2019_TS", "PGC_2020_PTSD", "PGC_2022_SCZ",
    "PGC_2024_BIP", "PGC_2025_BPD", "PGC_2025_MDD", "PGC_2025_OCD",
    "PGC_2026_GAD", "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE"
}
PRIMARY_22 = PRIMARY_NEURO | PRIMARY_PSYCH

ALL_PSYCH_KW = ["PGC_", "finngen_R11_F5_", "finngen_R11_KRA_PSY_"]
def get_domain(c):
    return "Psych" if any(k in str(c) for k in ALL_PSYCH_KW) else "Neuro"

SAME_DISORDER_GROUPS = [
    {"AD_2022_Bellenguez", "AD_2021_Wightman", "AD_2025_UKB_WGS"},
    {"ALS_2021_MinE_Full", "ALS_2021_MinE_Clin", "ALS_2024_Verma"},
    {"EPI_2023_ILAE_European", "EPI_2025_UKB_WGS"},
    {"MIG_2022_IHGC", "MIG_2025_UKB_WGS", "HEADACHE_Gen_Mbatchou", "HEADACHE_Ten_Jiang"},
    {"MS_2024_Science", "MS_2025_Shigesi", "MS_2025_UKB_WGS"},
    {"PD_2024_NatGen", "PD_2025_UKB_WGS"},
    {"PGC_2025_MDD", "finngen_R11_F5_DEPRESSIO"},
    {"PGC_2026_GAD", "finngen_R11_F5_ANXIETY"},
    {"PGC_2025_BPD", "finngen_R11_F5_EMOPER"},
    {"PGC_2024_BIP", "finngen_R11_F5_BIPO"},
    {"PGC_2025_OCD", "finngen_R11_F5_OCD"},
    {"PGC_2020_PTSD", "finngen_R11_F5_PTSD"},
    {"PGC_2022_SCZ", "finngen_R11_F5_SCHZPHR"},
    {"PGC_2019_AUT", "finngen_R11_KRA_PSY_AUTISM_EXMORE"},
    {"PGC_2023_SUD", "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE", "finngen_R11_F5_SUBSNOALCO"},
]
def is_same_disorder(t1, t2):
    if {t1, t2} == {"MIG_2022_IHGC", "HEADACHE_Gen_Mbatchou"}:
        return False
    return any(t1 in g and t2 in g for g in SAME_DISORDER_GROUPS)

def assign_tier(t1, t2):
    if is_same_disorder(t1, t2):
        return "Within_Disorder_Calibration"
    if t1 in PRIMARY_22 and t2 in PRIMARY_22:
        return "Primary_22_NonRedundant_Discovery"
    return "Cross_Biobank_Consistency_Benchmark"

# =============================================================================
# 2. DUSUK RAM'LI KOORDINAT VE 42 KOHORT NUMPY INDEKSLEMESI (~500 MB)
# =============================================================================
print("[1/5] GRCh38 HapMap3 koordinatlari okunuyor...", flush=True)
snp_to_idx = {}
idx_to_snp = []
chr_raw = defaultdict(list)

with gzip.open(COORD_FILE, "rt", encoding="utf-8", errors="ignore") as f:
    header = f.readline()  # KEY \t SNP
    for line in f:
        p = line.strip().split()
        if len(p) >= 2 and ":" in p[0]:
            ch_bp = p[0].split(":")
            c = ch_bp[0].replace("chr", "")
            try:
                bp = int(ch_bp[1])
            except ValueError:
                continue
            rs = p[1]
            if rs not in snp_to_idx:
                idx = len(idx_to_snp)
                snp_to_idx[rs] = idx
                idx_to_snp.append(rs)
                chr_raw[c].append((bp, idx))

# Aday dosyasindaki Lead_SNP'lerden eksik olan varsa ekle
df_cand = pd.read_csv(P6_CAND, sep="\t")
for _, r in df_cand[["Lead_SNP", "CHR", "BP_hg38"]].drop_duplicates().iterrows():
    rs = str(r["Lead_SNP"])
    if rs not in snp_to_idx:
        idx = len(idx_to_snp)
        snp_to_idx[rs] = idx
        idx_to_snp.append(rs)
        chr_raw[str(r["CHR"]).replace("chr", "")].append((int(r["BP_hg38"]), idx))

M = len(idx_to_snp)
chr_bps = {}
chr_idx = {}
for c, lst in chr_raw.items():
    lst.sort(key=lambda x: x[0])
    chr_bps[c] = np.array([x[0] for x in lst], dtype=np.int32)
    chr_idx[c] = np.array([x[1] for x in lst], dtype=np.int32)
del chr_raw
print(f"    -> {M:,} GRCh38 SNP indekslendi ({time.time()-t0:.1f}s).", flush=True)

# Sadece adaylarda gecen kohortlari yukle (bellek + hiz tasarrufu)
needed_cohorts = sorted(set(df_cand["Trait_1"]).union(set(df_cand["Trait_2"])))
ALLELE_CODE = {"A": 1, "C": 2, "G": 3, "T": 4}
sumstats_map = {
    os.path.basename(sf).replace("_Ready.sumstats.gz", ""): sf
    for sf in glob.glob(os.path.join(SUMSTATS_DIR, "*_Ready.sumstats.gz"))
}

print(f"[2/5] {len(needed_cohorts)} kohort kompakt NumPy dizilerine yukleniyor...", flush=True)
cohort_z = {}
cohort_n = {}
cohort_a1 = {}
cohort_a2 = {}

for ci, cname in enumerate(needed_cohorts, 1):
    sf = sumstats_map.get(cname)
    if not sf:
        continue
    z_arr = np.full(M, np.nan, dtype=np.float32)
    n_arr = np.full(M, 100000.0, dtype=np.float32)
    a1_arr = np.zeros(M, dtype=np.uint8)
    a2_arr = np.zeros(M, dtype=np.uint8)
    with gzip.open(sf, "rt", encoding="utf-8", errors="ignore") as f:
        hdr = f.readline().strip().split()
        hm = {col.upper(): i for i, col in enumerate(hdr)}
        si, a1i, a2i, zi, ni = hm.get("SNP", 0), hm.get("A1", 1), hm.get("A2", 2), hm.get("Z", 3), hm.get("N", 4)
        for line in f:
            p = line.strip().split()
            if len(p) > zi:
                idx = snp_to_idx.get(p[si])
                if idx is not None:
                    try:
                        zv = float(p[zi])
                        if not (math.isnan(zv) or math.isinf(zv)):
                            z_arr[idx] = zv
                            a1_arr[idx] = ALLELE_CODE.get(p[a1i].upper(), 0)
                            a2_arr[idx] = ALLELE_CODE.get(p[a2i].upper(), 0)
                            if len(p) > ni:
                                n_arr[idx] = float(p[ni])
                    except ValueError:
                        pass
    cohort_z[cname] = z_arr
    cohort_n[cname] = n_arr
    cohort_a1[cname] = a1_arr
    cohort_a2[cname] = a2_arr
    if ci % 10 == 0 or ci == len(needed_cohorts):
        print(f"    -> {ci}/{len(needed_cohorts)} kohort yuklendi ({time.time()-t0:.1f}s)", flush=True)

# =============================================================================
# 3. 17q21.31 (43.5-47.0 Mb) KONSOLIDASYON + SIMETRIK PLACO + GERCEK COLOC
# =============================================================================
print("[3/5] 17q21.31 konsolidasyonu, simetrik ZCA PLACO ve gercek coklu-SNP COLOC hesaplaniyor...", flush=True)
df_p11 = pd.read_csv(P11_FILE, sep="\t")
annot_cols = [c for c in ["Resolved_Gene", "VEP_Consequence_Resolved", "CADD_PHRED_Resolved"] if c in df_p11.columns]
annot_map = {(str(r["Pair_Name"]), str(r["Lead_SNP"])): {c: r[c] for c in annot_cols} for _, r in df_p11.iterrows()}

def assign_ld_block(row):
    ch = str(row["CHR"]).replace("chr", "")
    bp = int(row["BP_hg38"])
    if ch == "6" and 25000000 <= bp <= 34000000:
        return "EXTENDED_MHC_CHR6"
    if ch == "17" and 43500000 <= bp <= 47000000:
        return "CHR17Q21_31_MAPT_CRHR1_WNT3_NSF"
    return f"STD_CHR{ch}_{bp // 500000}"

df_cand["LD_Block_Clean"] = df_cand.apply(assign_ld_block, axis=1)
df_cand["Abs_Prod"] = df_cand["Z_Trait1"].abs() * df_cand["Z_Trait2"].abs()
before_dedup = len(df_cand)
df_cand = df_cand.sort_values("Abs_Prod", ascending=False).drop_duplicates(subset=["Pair_Name", "LD_Block_Clean"], keep="first").reset_index(drop=True)
print(f"    -> 17q21.31 (43.5-47.0 Mb) / MHC konsolidasyonu: {before_dedup} -> {len(df_cand)} aday lokus.", flush=True)

# Vektorize alel hizalama
def align_pair_vectors(t1, t2, idxs=None):
    if idxs is None:
        z1, z2 = cohort_z[t1], cohort_z[t2]
        a1_1, a2_1 = cohort_a1[t1], cohort_a2[t1]
        a1_2, a2_2 = cohort_a1[t2], cohort_a2[t2]
    else:
        z1, z2 = cohort_z[t1][idxs], cohort_z[t2][idxs]
        a1_1, a2_1 = cohort_a1[t1][idxs], cohort_a2[t1][idxs]
        a1_2, a2_2 = cohort_a1[t2][idxs], cohort_a2[t2][idxs]
    ok = (~np.isnan(z1)) & (~np.isnan(z2)) & (a1_1 > 0) & (a2_1 > 0) & (a1_2 > 0) & (a2_2 > 0)
    same = (a1_1 == a1_2) & (a2_1 == a2_2)
    flip = (a1_1 == a2_2) & (a2_1 == a1_2)
    c1 = 5 - a1_2
    c2 = 5 - a2_2
    csame = (a1_1 == c1) & (a2_1 == c2)
    cflip = (a1_1 == c2) & (a2_1 == c1)
    valid = ok & (same | flip | csame | cflip)
    sign = np.where(flip | cflip, -1.0, 1.0).astype(np.float32)
    return valid, z1[valid].astype(np.float64), (z2[valid] * sign[valid]).astype(np.float64)

# Her benzersiz cift icin genom-geneli kanonik PLACO (Ray & Chatterjee 2020: P>1e-4 r_null, Z^2<=80 VarZ)
pair_params = {}
unique_pairs = df_cand[["Trait_1", "Trait_2", "Pair_Name"]].drop_duplicates()
for _, pr in unique_pairs.iterrows():
    t1, t2, pname = pr["Trait_1"], pr["Trait_2"], pr["Pair_Name"]
    if t1 not in cohort_z or t2 not in cohort_z:
        pair_params[pname] = (0.0, 1.0, 1.0)
        continue
    _, z1_v, z2_v = align_pair_vectors(t1, t2)
    if len(z1_v) < 1000:
        pair_params[pname] = (0.0, 1.0, 1.0)
        continue
    # Kanonik PLACO cor.pearson (p.threshold = 1e-4 => |Z| < 3.8906, kesilme zayiflamasi yok)
    null_m = (np.abs(z1_v) < 3.8906) & (np.abs(z2_v) < 3.8906)
    r_null = float(np.clip(np.corrcoef(z1_v[null_m], z2_v[null_m])[0, 1], -0.85, 0.85)) if null_m.sum() > 500 else 0.0
    # Simetrik ZCA whitening R^(-1/2)
    a = 0.5 * (1.0 / math.sqrt(1.0 + r_null) + 1.0 / math.sqrt(1.0 - r_null))
    b = 0.5 * (1.0 / math.sqrt(1.0 + r_null) - 1.0 / math.sqrt(1.0 - r_null))
    # Z^2 <= 80 filtresi (Ray & Chatterjee 2020)
    m80 = (z1_v**2 <= 80.0) & (z2_v**2 <= 80.0)
    z1_w = a * z1_v[m80] + b * z2_v[m80]
    z2_w = b * z1_v[m80] + a * z2_v[m80]
    v1 = max(1.0, float(np.var(z1_w)))
    v2 = max(1.0, float(np.var(z2_w)))
    pair_params[pname] = (r_null, v1, v2)

print(f"    -> {len(pair_params)} cift icin genom-geneli ZCA parametreleri tamamlandi ({time.time()-t0:.1f}s).", flush=True)

def normal_prod_tail(t):
    t = float(abs(t))
    if t <= 0:
        return 1.0
    if t > 500.0:
        return max(1e-300, math.sqrt(2.0 / (math.pi * t)) * math.exp(-t))
    val, _ = quad(lambda u: kn(0, u), t, np.inf, limit=80)
    return min(1.0, max(1e-300, (2.0 / math.pi) * val))

def canonical_placo_p(z1s, z2s, var1, var2):
    T = abs(z1s * z2s)
    p = normal_prod_tail(T / math.sqrt(var1)) + normal_prod_tail(T / math.sqrt(var2)) - normal_prod_tail(T)
    return min(1.0, max(1e-300, p))

W_PRIOR = 0.04   # Standart Wakefield log-OR oncel varyansi (0.2^2)
P1_PR, P2_PR, P12_PR = 1e-4, 1e-4, 5e-5

out_rows = []
for _, row in df_cand.iterrows():
    t1, t2, pname = row["Trait_1"], row["Trait_2"], row["Pair_Name"]
    lead = str(row["Lead_SNP"])
    ch = str(row["CHR"]).replace("chr", "")
    bp = int(row["BP_hg38"])
    l_idx = snp_to_idx.get(lead)

    # Lead SNP alel-hizali ham Z skorlari
    if l_idx is not None and t1 in cohort_z and t2 in cohort_z:
        valid_l, z1_l, z2_l = align_pair_vectors(t1, t2, np.array([l_idx], dtype=np.int32))
        if valid_l[0]:
            z1_raw, z2_al, al_stat = float(z1_l[0]), float(z2_l[0]), "Aligned"
        else:
            z1_raw, z2_al, al_stat = float(row["Z_Trait1"]), float(row["Z_Trait2"]), "Fallback"
    else:
        z1_raw, z2_al, al_stat = float(row["Z_Trait1"]), float(row["Z_Trait2"]), "Missing"

    # Kanonik simetrik ZCA PLACO
    r_null, v1, v2 = pair_params.get(pname, (0.0, 1.0, 1.0))
    a = 0.5 * (1.0 / math.sqrt(1.0 + r_null) + 1.0 / math.sqrt(1.0 - r_null))
    b = 0.5 * (1.0 / math.sqrt(1.0 + r_null) - 1.0 / math.sqrt(1.0 - r_null))
    z1_w = a * z1_raw + b * z2_al
    z2_w = b * z1_raw + a * z2_al
    p_placo_sym = canonical_placo_p(z1_w, z2_w, v1, v2)

    # Gercek Bolgesel Coklu-SNP COLOC (+/- 250 kb, TUM ortak HapMap3 SNP'leri!)
    bps = chr_bps.get(ch)
    idxs = chr_idx.get(ch)
    if bps is not None and t1 in cohort_z and t2 in cohort_z:
        li = int(np.searchsorted(bps, bp - 250000, side="left"))
        ri = int(np.searchsorted(bps, bp + 250000, side="right"))
        win_idxs = idxs[li:ri]
        valid_w, z1_v, z2_v = align_pair_vectors(t1, t2, win_idxs)
        n_reg = int(valid_w.sum())
    else:
        n_reg = 0

    if n_reg >= 5:
        kept_idxs = win_idxs[valid_w]
        n1_v = np.maximum(1000.0, cohort_n[t1][kept_idxs].astype(np.float64))
        n2_v = np.maximum(1000.0, cohort_n[t2][kept_idxs].astype(np.float64))
        V1 = 1.0 / (n1_v * 0.375)
        V2 = 1.0 / (n2_v * 0.375)
        r1 = W_PRIOR / (W_PRIOR + V1)
        r2 = W_PRIOR / (W_PRIOR + V2)
        lABF1 = 0.5 * (np.log(1.0 - r1) + r1 * (z1_v ** 2))
        lABF2 = 0.5 * (np.log(1.0 - r2) + r2 * (z2_v ** 2))
        lABF12 = lABF1 + lABF2

        ls1 = float(logsumexp(lABF1))
        ls2 = float(logsumexp(lABF2))
        ls12 = float(logsumexp(lABF12))
        lH0 = 0.0
        lH1 = math.log(P1_PR) + ls1
        lH2 = math.log(P2_PR) + ls2
        lH4 = math.log(P12_PR) + ls12
        sum_both = ls1 + ls2
        if sum_both > ls12 + 1e-12:
            lH3 = math.log(P1_PR) + math.log(P2_PR) + sum_both + math.log(max(1e-300, 1.0 - math.exp(ls12 - sum_both)))
        else:
            lH3 = -1e9

        denom = float(logsumexp([lH0, lH1, lH2, lH3, lH4]))
        pp_h3 = math.exp(lH3 - denom)
        pp_h4 = math.exp(lH4 - denom)

        post_snp = np.exp(lABF12 - ls12)
        order_s = np.argsort(-post_snp)
        cum = 0.0
        cs_size = 0
        for oi in order_s:
            cum += post_snp[oi]
            cs_size += 1
            if cum >= 0.95:
                break
        top_snp_pp = float(post_snp[order_s[0]])
        top_cs_snp = idx_to_snp[kept_idxs[order_s[0]]]
    else:
        pp_h4, pp_h3, cs_size, top_snp_pp, top_cs_snp = 0.0, 0.0, 0, 0.0, lead

    direction = "Concordant (+/+)" if (z1_raw * z2_al) > 0 else "Discordant (+/-)"
    d_t1, d_t2 = get_domain(t1), get_domain(t2)
    dom_pair = "Cross_Neuro_Psych" if d_t1 != d_t2 else f"Intra_{d_t1}"
    tier = assign_tier(t1, t2)
    p11_info = annot_map.get((pname, lead), {})
    res_gene = p11_info.get("Resolved_Gene", row.get("Mapped_Gene", "Unmapped"))

    out_rows.append({
        "Trait_1": t1, "Trait_2": t2, "Pair_Name": pname,
        "Pair_Tier": tier, "Domain_Pair": dom_pair,
        "Lead_SNP": lead, "CHR": ch, "BP_hg38": bp, "LD_Block_Clean": row["LD_Block_Clean"],
        "Z_Trait1": round(z1_raw, 4), "Z_Trait2_Aligned": round(z2_al, 4),
        "Allele_Align_Status": al_stat, "Direction_Aligned": direction,
        "Novelty_Class": row.get("Novelty_Class", ""),
        "r_null_Empirical": round(r_null, 4), "PLACO_Var1": round(v1, 4), "PLACO_Var2": round(v2, 4),
        "P_PLACO_Sym": p_placo_sym,
        "Regional_SNP_Count": n_reg, "COLOC_PP_H4_Real": round(pp_h4, 5), "COLOC_PP_H3_Real": round(pp_h3, 5),
        "CS95_Size_Real": cs_size, "Top_SNP_Cond_PP_H4": round(top_snp_pp, 4), "Top_CS95_SNP": top_cs_snp,
        "Resolved_Gene": res_gene,
        "VEP_Consequence": p11_info.get("VEP_Consequence_Resolved", row.get("Consequence", "")),
        "CADD_PHRED": p11_info.get("CADD_PHRED_Resolved", row.get("CADD_PHRED", np.nan)),
    })

df_all = pd.DataFrame(out_rows)
df_all.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_All_Candidates_RealPLACO_RealCOLOC.tsv"), sep="\t", index=False)

pass_placo = df_all["P_PLACO_Sym"] < 5e-8
pass_coloc = df_all["COLOC_PP_H4_Real"] >= 0.75
df_val = df_all[pass_placo & pass_coloc].copy()
df_val.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Validated_All861.tsv"), sep="\t", index=False)

df_prim = df_val[df_val["Pair_Tier"] == "Primary_22_NonRedundant_Discovery"].copy()
df_prim.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t", index=False)

print(f"    -> PLACO + COLOC tamamlandi ({time.time()-t0:.1f}s).\n", flush=True)

print("=" * 95)
print(" [OZET 1] GERCEK COKLU-SNP COLOC + SIMETRIK PLACO HUNISI (0.9000 HATASI SIFIRLANDI)")
print("=" * 95)
print(f"  Toplam konsolide aday lokus (861 cift) : {len(df_all)}")
print(f"  Bolgesel SNP sayisi medyani            : {df_all['Regional_SNP_Count'].median():.0f} (min={df_all['Regional_SNP_Count'].min()}, max={df_all['Regional_SNP_Count'].max()})")
print(f"  PP.H4 == 0.9000 sayisi (eski: 898!)    : {(df_val['COLOC_PP_H4_Real'] == 0.9000).sum()}")
print(f"  Gecen COLOC (PP.H4 >= 0.75)            : {int(pass_coloc.sum())}")
print(f"  Gecen Simetrik PLACO (P < 5e-8)        : {int(pass_placo.sum())}")
print(f"  IKISINI BIRDEN GECEN (861 cift toplam) : {len(df_val)}")
print(f"  Tier kirilimi:\n{df_val['Pair_Tier'].value_counts().to_string()}")

for label, sub in [("TUM 861 CIFT", df_val), ("BIRINCIL TEKRARSIZ 22 BOZUKLUK (231 CIFT)", df_prim)]:
    genes = sub["Resolved_Gene"].dropna().astype(str)
    hgnc = genes[~genes.str.startswith("ENSG") & ~genes.str.startswith("LOC_chr")]
    cross = sub[sub["Domain_Pair"] == "Cross_Neuro_Psych"]
    intra_p = sub[sub["Domain_Pair"] == "Intra_Psych"]
    c_disc = int((cross["Direction_Aligned"] == "Discordant (+/-)").sum())
    ip_conc = int((intra_p["Direction_Aligned"] == "Concordant (+/+)").sum())
    nov = int((sub["Novelty_Class"] == "HIDDEN_NOVEL_PLEIOTROPY").sum())
    cs1 = int((sub["CS95_Size_Real"] == 1).sum())
    print(f"\n  --- {label} ---")
    print(f"    Lokus: {len(sub)} | Benzersiz Cift: {sub['Pair_Name'].nunique()} | Gen (Tumu): {genes.nunique()} | HGNC: {hgnc.nunique()}")
    print(f"    Subthreshold (Hidden Novel): {nov} ({100*nov/max(1,len(sub)):.1f}%) | CS95 medyan: {sub['CS95_Size_Real'].median():.0f} | CS95=1: {cs1} ({100*cs1/max(1,len(sub)):.1f}%)")
    print(f"    Genel (+/+): {(sub['Direction_Aligned']=='Concordant (+/+)').sum()} ({100*(sub['Direction_Aligned']=='Concordant (+/+)').mean():.1f}%)")
    print(f"    Intra_Psych (n={len(intra_p)}): (+/+) = {ip_conc} ({100*ip_conc/max(1,len(intra_p)):.1f}%)")
    print(f"    Cross_Neuro_Psych (n={len(cross)}): (+/-) = {c_disc} ({100*c_disc/max(1,len(cross)):.1f}%)")
    b231_h90 = cross[(cross["P_PLACO_Sym"] < 5e-8/231.0) & (cross["COLOC_PP_H4_Real"] >= 0.90)]
    b861_h90 = cross[(cross["P_PLACO_Sym"] < 5e-8/861.0) & (cross["COLOC_PP_H4_Real"] >= 0.90)]
    if len(b231_h90) > 0:
        print(f"    Cross Bonf-231 + PP.H4>=0.90: n={len(b231_h90)} | (+/-) = {100*(b231_h90['Direction_Aligned']=='Discordant (+/-)').mean():.1f}%")
    if len(b861_h90) > 0:
        print(f"    Cross Bonf-861 + PP.H4>=0.90: n={len(b861_h90)} | (+/-) = {100*(b861_h90['Direction_Aligned']=='Discordant (+/-)').mean():.1f}%")

# =============================================================================
# 4. BIRINCIL TEKRARSIZ HUB GENLER (TABLE 3)
# =============================================================================
print("\n[4/5] Birincil 22-bozukluk Top 15 Hub Geni hesaplaniyor...", flush=True)
prim_hgnc = df_prim[~df_prim["Resolved_Gene"].astype(str).str.startswith(("ENSG", "LOC_chr"))].copy()
hub_rows = []
for g, grp in prim_hgnc.groupby("Resolved_Gene"):
    grp_s = grp.sort_values("P_PLACO_Sym")
    top = grp_s.iloc[0]
    hub_rows.append({
        "Gene": g,
        "Unique_Primary_Pairs": grp["Pair_Name"].nunique(),
        "Cross_Domain_Pairs": grp[grp["Domain_Pair"] == "Cross_Neuro_Psych"]["Pair_Name"].nunique(),
        "Min_P_PLACO": f"{top['P_PLACO_Sym']:.2e}",
        "Median_PP_H4": round(float(grp["COLOC_PP_H4_Real"].median()), 4),
        "Lead_CS95": int(top["CS95_Size_Real"]),
        "Lead_Cond_PP": round(float(top["Top_SNP_Cond_PP_H4"]), 4),
        "Lead_SNP": top["Lead_SNP"],
        "Rep_Pairs": "; ".join(grp_s["Pair_Name"].unique()[:3])
    })
df_hubs = pd.DataFrame(hub_rows).sort_values(["Unique_Primary_Pairs", "Cross_Domain_Pairs"], ascending=False)
df_hubs.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Table3_Primary22_HubGenes.tsv"), sep="\t", index=False)
print(df_hubs.head(15)[["Gene", "Unique_Primary_Pairs", "Cross_Domain_Pairs", "Min_P_PLACO", "Median_PP_H4", "Lead_CS95", "Lead_Cond_PP"]].to_string(index=False))

# =============================================================================
# 5. LDSC BIRINCIL 22x22 MATRIS, PURE-EUR DUYARLILIK, HATASIZ FIGURE 1 & TABLE 2
# =============================================================================
print("\n[5/5] LDSC Birincil 22x22 BH-FDR, Hatasiz Figure 1 ve Temiz Table 2 uretiliyor...", flush=True)
df_ldsc = pd.read_csv(LDSC_FILE, sep="\t")
df_ldsc["Pair_Tier"] = df_ldsc.apply(lambda r: assign_tier(r["Trait_1"], r["Trait_2"]), axis=1)
df_ldsc["D1"] = df_ldsc["Trait_1"].apply(get_domain)
df_ldsc["D2"] = df_ldsc["Trait_2"].apply(get_domain)
df_ldsc["Domain_Pair"] = df_ldsc.apply(lambda r: "Cross_Neuro_Psych" if r["D1"] != r["D2"] else f"Intra_{r['D1']}", axis=1)

prim_ldsc = df_ldsc[df_ldsc["Pair_Tier"] == "Primary_22_NonRedundant_Discovery"].dropna(subset=["p"]).reset_index(drop=True)
m = len(prim_ldsc)
order_p = np.argsort(prim_ldsc["p"].values)
p_sorted = prim_ldsc["p"].values[order_p]
fdr_sorted = np.clip(np.minimum.accumulate((p_sorted * m / np.arange(1, m + 1))[::-1])[::-1], 0.0, 1.0)
fdr_vals = np.empty(m)
fdr_vals[order_p] = fdr_sorted
prim_ldsc["FDR_Primary231"] = fdr_vals
prim_ldsc.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Primary22_LDSC_231Pairs.tsv"), sep="\t", index=False)

sig_prim = prim_ldsc[prim_ldsc["FDR_Primary231"] < 0.05]
print(f"\n  Birincil 22x22 LDSC cift: {len(prim_ldsc)} / 231 | BH-FDR < 0.05: {len(sig_prim)}")
print(sig_prim["Domain_Pair"].value_counts().to_string())

top_cross_ldsc = sig_prim[sig_prim["Domain_Pair"] == "Cross_Neuro_Psych"].sort_values("p")
top_cross_ldsc.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Table1_Primary22_TopCrossLDSC.tsv"), sep="\t", index=False)
print("\n  Top 15 Birincil Cross-Domain LDSC Korelasyonu (P-degerine gore sirali):")
print(top_cross_ldsc[["Trait_1", "Trait_2", "rg", "se", "z", "p", "FDR_Primary231"]].head(15).to_string(index=False))

# Pure-EUR Sensitivity
df_qc = pd.read_csv(QC_FILE, sep="\t")
pure_eur = set(df_qc[(df_qc["Ancestry"] == "European (EUR)") & (df_qc["LDSC_h2_Z"] >= 4.0) & (df_qc["N_HM3_SNPs"] >= 750000)]["Cohort"])
eur_prim = prim_ldsc[prim_ldsc["Trait_1"].isin(pure_eur) & prim_ldsc["Trait_2"].isin(pure_eur)]
print(f"\n  Pure-EUR yuksek guclu birincil kohort: {len(pure_eur & PRIMARY_22)} | Cift: {len(eur_prim)} | FDR<0.05: {(eur_prim['FDR_Primary231']<0.05).sum()}")

# HATASIZ FIGURE 1 (22x22)
cohorts_22 = sorted(list(PRIMARY_22))
n22 = len(cohorts_22)
c_idx = {c: i for i, c in enumerate(cohorts_22)}
rg_mat = np.zeros((n22, n22))
fdr_mat = np.ones((n22, n22))
np.fill_diagonal(rg_mat, 1.0)
for _, r in prim_ldsc.iterrows():
    i, j = c_idx[r["Trait_1"]], c_idx[r["Trait_2"]]
    val = float(np.clip(r["rg"], -1.0, 1.0)) if pd.notna(r["rg"]) else 0.0
    rg_mat[i, j] = rg_mat[j, i] = val
    fdr_mat[i, j] = fdr_mat[j, i] = float(r["FDR_Primary231"])

dist_mat = np.clip(1.0 - rg_mat, 0.0, 2.0)
np.fill_diagonal(dist_mat, 0.0)
order = leaves_list(linkage(squareform(dist_mat), method="ward"))
rg_s = rg_mat[order][:, order]
fdr_s = fdr_mat[order][:, order]
CLEAN_NAMES = {
    "AD_2022_Bellenguez": "Alzheimer's (Bellenguez)", "ALS_2021_MinE_Full": "ALS (Project MinE)",
    "EPI_2023_ILAE_European": "Epilepsy (ILAE)", "ET_2024_Verma": "Essential Tremor (MVP)",
    "HBS_2024_Annals": "Brain Structure / HBS", "HEADACHE_Gen_Mbatchou": "Broad Headache (UKB)",
    "MG_2024_Braun": "Myasthenia Gravis", "MIG_2022_IHGC": "Migraine (IHGC)",
    "MS_2024_Science": "Multiple Sclerosis (IMSGC)", "NARCO_2024_Verma": "Narcolepsy (MVP)",
    "PD_2024_NatGen": "Parkinson's (IPDGC)", "STROKE_2022_GIGA": "Ischemic Stroke (GIGA)",
    "PGC_2019_AUT": "Autism (PGC)", "PGC_2019_TS": "Tourette Syndrome (PGC)",
    "PGC_2020_PTSD": "PTSD (PGC)", "PGC_2022_SCZ": "Schizophrenia (PGC)",
    "PGC_2024_BIP": "Bipolar Disorder (PGC)", "PGC_2025_BPD": "Borderline PD (PGC)",
    "PGC_2025_MDD": "Major Depression (PGC)", "PGC_2025_OCD": "OCD (PGC)",
    "PGC_2026_GAD": "Anxiety / GAD (PGC)", "finngen_R11_KRA_PSY_SUBSTANCE_EXMORE": "Substance Use (FinnGen)"
}
lbl_s = [CLEAN_NAMES.get(cohorts_22[i], cohorts_22[i]) for i in order]

fig, ax = plt.subplots(figsize=(11.5, 10.2), dpi=300)
im = ax.imshow(rg_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
ax.set_xticks(np.arange(n22))
ax.set_yticks(np.arange(n22))
ax.set_xticklabels(lbl_s, rotation=45, ha="right", fontsize=8.8, fontweight="bold")
ax.set_yticklabels(lbl_s, fontsize=8.8, fontweight="bold")
for i in range(n22):
    for j in range(n22):
        if i != j and fdr_s[i, j] < 0.05:
            ax.text(j, i, "*", ha="center", va="center",
                    color="white" if abs(rg_s[i, j]) >= 0.45 else "black", fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Global Genetic Correlation ($r_g$)", fontweight="bold")
ax.set_title("Primary Non-Redundant Global Genetic Correlation Matrix (22 Unique Disorders, * BH-FDR < 0.05)", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure1_Primary22_NonRedundant_Heatmap.pdf"))
plt.savefig(os.path.join(OUT_DIR, "Figure1_Primary22_NonRedundant_Heatmap.png"), dpi=300)
plt.close()

# HATASIZ TABLE 2: 5 FAKTOR (ORTUSMESIZ 10 PSIKIYATRI) x 4 NOROLOJI (12 NOROLOJI) = 120 BIRINCIL CAPRAZ CIFT
PSYCH_FACTORS = {
    "F1: Compulsive (OCD)": {"PGC_2025_OCD"},
    "F2: Schizophrenia/Bipolar (SCZ, BIP)": {"PGC_2022_SCZ", "PGC_2024_BIP"},
    "F3: Neurodevelopmental (ASD, TS*)": {"PGC_2019_AUT", "PGC_2019_TS"},
    "F4: Internalizing (MDD, GAD, PTSD + BPD ext.)": {"PGC_2025_MDD", "PGC_2026_GAD", "PGC_2020_PTSD", "PGC_2025_BPD"},
    "F5: Substance Use (SUD)": {"finngen_R11_KRA_PSY_SUBSTANCE_EXMORE"},
}
NEURO_CLASSES = {
    "Neurovascular & Headache (STROKE, MIG, HEADACHE, HBS)": {"STROKE_2022_GIGA", "MIG_2022_IHGC", "HEADACHE_Gen_Mbatchou", "HBS_2024_Annals"},
    "Movement & Cerebellar (ET, PD)": {"ET_2024_Verma", "PD_2024_NatGen"},
    "Neurodegenerative (AD, ALS)": {"AD_2022_Bellenguez", "ALS_2021_MinE_Full"},
    "Neuroimmune & Excitability (MS, MG, EPI, NARCO)": {"MS_2024_Science", "MG_2024_Braun", "EPI_2023_ILAE_European", "NARCO_2024_Verma"},
}

t2_rows = []
tot_l, tot_c, tot_d, tot_sig, tot_p = 0, 0, 0, 0, 0
for fn, fset in PSYCH_FACTORS.items():
    for nc, nset in NEURO_CLASSES.items():
        m_ldsc = prim_ldsc[
            (prim_ldsc["Trait_1"].isin(fset) & prim_ldsc["Trait_2"].isin(nset)) |
            (prim_ldsc["Trait_2"].isin(fset) & prim_ldsc["Trait_1"].isin(nset))
        ]
        n_tot_p = len(m_ldsc)
        n_sig_p = int((m_ldsc["FDR_Primary231"] < 0.05).sum())
        mean_all_rg = float(m_ldsc["rg"].mean()) if n_tot_p > 0 else float("nan")
        max_all_rg = float(m_ldsc["rg"].max()) if n_tot_p > 0 else float("nan")

        m_loc = df_prim[
            (df_prim["Trait_1"].isin(fset) & df_prim["Trait_2"].isin(nset)) |
            (df_prim["Trait_2"].isin(fset) & df_prim["Trait_1"].isin(nset))
        ]
        n_loc = len(m_loc)
        n_uniq_blocks = m_loc["LD_Block_Clean"].nunique()
        n_conc = int((m_loc["Direction_Aligned"] == "Concordant (+/+)").sum())
        n_disc = int((m_loc["Direction_Aligned"] == "Discordant (+/-)").sum())
        tot_l += n_loc; tot_c += n_conc; tot_d += n_disc; tot_sig += n_sig_p; tot_p += n_tot_p
        top_g = m_loc[~m_loc["Resolved_Gene"].astype(str).str.startswith(("ENSG", "LOC_chr"))]["Resolved_Gene"].value_counts().head(4).index.tolist()
        t2_rows.append({
            "Factor": fn, "Neuro_Class": nc,
            "FDR_Sig_Pairs": f"{n_sig_p} / {n_tot_p}",
            "Mean_All_rg (Max)": f"{mean_all_rg:.3f} ({max_all_rg:.3f})",
            "Loci (Unique_Regions)": f"{n_loc} ({n_uniq_blocks})",
            "Conc_vs_Disc": f"{n_conc} (+/+) vs. {n_disc} (+/-)",
            "Top_Genes": ", ".join(top_g) if top_g else "—"
        })

df_t2 = pd.DataFrame(t2_rows)
df_t2.to_csv(os.path.join(OUT_DIR, "CEREBRAX_Phase14_Table2_5Factors_Clean.tsv"), sep="\t", index=False)
print("\n" + "=" * 95)
print(f" [OZET 4] HATASIZ TABLE 2 (SATIR TOPLAMI: {tot_sig}/{tot_p} FDR cift | {tot_l} lokus: {tot_c} +/+ vs {tot_d} +/-)")
print("=" * 95)
print(df_t2.to_string(index=False))
print(f"\nTAMAMLANDI ({time.time()-t0:.1f}s) -> {OUT_DIR}")
