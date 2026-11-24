# -*- coding: utf-8 -*-
import os, gzip, glob, re, json, math, time
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, binomtest, norm, chi2, t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
BASE = "/home/cihat/CEREBRAX_PROJECT"
P14 = os.path.join(BASE, "phase14_bulletproof")
SUMSTATS_DIR = os.path.join(BASE, "data", "munged_data")
COORD_FILE = os.path.join(BASE, "ref_data", "hm3_hg38_chr_bp_to_rsid.tsv.gz")

df_all_cand = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_All_Candidates_RealPLACO_RealCOLOC.tsv"), sep="\t")
df_val = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Validated_All861.tsv"), sep="\t")
df_prim = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Primary22_NonRedundant.tsv"), sep="\t")
df_hubs = pd.read_csv(os.path.join(P14, "CEREBRAX_Phase14_Table3_Primary22_HubGenes.tsv"), sep="\t")

# =============================================================================
# 1. CAPRAZ-BIYOBANKA TUTARLILIK, TEKIL SNP REPLIKASYONU & LDSC gcov_int
# =============================================================================
print("[1/4] Capraz-biyobanka tutarlilik (tekil SNP + |Z|) ve LDSC gcov_int hesaplaniyor...", flush=True)

# LDSC loglarindan bivariate gcov_int (orneklem cakismasi) cek
gcov_map = {}
for lf in glob.glob(os.path.join(BASE, "**", "*.log"), recursive=True):
    try:
        txt = open(lf, errors="ignore").read()
    except Exception:
        continue
    m = re.search(r"Summary of Genetic Correlation Results\s*\n[^\n]+\n([^\n]+)", txt)
    if m:
        parts = m.group(1).strip().split()
        if len(parts) >= 12:
            c1 = os.path.basename(parts[0]).replace("_Ready.sumstats.gz", "").replace(".sumstats.gz", "")
            c2 = os.path.basename(parts[1]).replace("_Ready.sumstats.gz", "").replace(".sumstats.gz", "")
            try:
                gcov_map[tuple(sorted([c1, c2]))] = (float(parts[10]), float(parts[11]))
            except ValueError:
                pass

# Eslesmis Primary -> Benchmark/FinnGen kohort ciftleri (orneklem cakismasi + replikasyon)
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

print("\n  Kesif vs. Tutarlilik Kohortlari Arasi Bivariate LDSC Intercept (Orneklem Cakismasi):")
ov_rows = []
for p_c, r_c in REP_MAP.items():
    gi, gse = gcov_map.get(tuple(sorted([p_c, r_c])), (float("nan"), float("nan")))
    ov_rows.append({"Primary_Cohort": p_c, "Benchmark_Cohort": r_c, "LDSC_gcov_int": gi, "SE": gse})
df_ov = pd.DataFrame(ov_rows)
df_ov.to_csv(os.path.join(P14, "CEREBRAX_Phase14_SampleOverlap_gcov_int.tsv"), sep="\t", index=False)
print(df_ov.to_string(index=False))

# Replikasyon: Sadece Primary 22'de kesfedilen lead SNP'leri dogrudan karsilik gelen Benchmark sumstats'ta ara
# (On-elemesiz! Sadece kesif lokusunun karsilik gelen kohorttaki ham Z-skoru!)
needed_rep_snps = set(df_prim["Lead_SNP"].astype(str))
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
def align_single(a1_1, a2_1, a1_2, a2_2, z2):
    if a1_1 == a1_2 and a2_1 == a2_2: return z2
    if a1_1 == a2_2 and a2_1 == a1_2: return -z2
    c1, c2 = COMP.get(a1_2, ""), COMP.get(a2_2, "")
    if a1_1 == c1 and a2_1 == c2: return z2
    if a1_1 == c2 and a2_1 == c1: return -z2
    return None

# Sadece lead SNP'ler icin tum ilgili sumstats'i hizlica oku (~5 saniye)
lead_db = {}
target_c = set(REP_MAP.keys()) | set(REP_MAP.values())
for cname in sorted(target_c):
    sf = os.path.join(SUMSTATS_DIR, f"{cname}_Ready.sumstats.gz")
    if not os.path.exists(sf): continue
    d = {}
    with gzip.open(sf, "rt", encoding="utf-8", errors="ignore") as f:
        hdr = f.readline().strip().split()
        hm = {col.upper(): i for i, col in enumerate(hdr)}
        si, a1i, a2i, zi = hm.get("SNP", 0), hm.get("A1", 1), hm.get("A2", 2), hm.get("Z", 3)
        for line in f:
            p = line.strip().split()
            if len(p) > zi and p[si] in needed_rep_snps:
                try:
                    d[p[si]] = (p[a1i].upper(), p[a2i].upper(), float(p[zi]))
                except ValueError:
                    pass
    lead_db[cname] = d

rep_records = []
seen_snp_cohort = set()
for _, row in df_prim.iterrows():
    rs = str(row["Lead_SNP"])
    for tc in [row["Trait_1"], row["Trait_2"]]:
        if tc in REP_MAP:
            rc = REP_MAP[tc]
            key = (rs, tc, rc)
            if key in seen_snp_cohort:
                continue
            seen_snp_cohort.add(key)
            if rs in lead_db.get(tc, {}) and rs in lead_db.get(rc, {}):
                a1_1, a2_1, z1 = lead_db[tc][rs]
                a1_2, a2_2, z2 = lead_db[rc][rs]
                z2a = align_single(a1_1, a2_1, a1_2, a2_2, z2)
                if z2a is not None and not (math.isnan(z1) or math.isnan(z2a)):
                    rep_records.append({
                        "Lead_SNP": rs, "Primary_Cohort": tc, "Benchmark_Cohort": rc,
                        "Z_Disc": z1, "Z_Rep_Aligned": z2a,
                        "Same_Sign": (z1 * z2a) > 0, "Nom_Sig": abs(z2a) >= 1.96
                    })

df_rep = pd.DataFrame(rep_records)
df_rep.to_csv(os.path.join(P14, "CEREBRAX_Phase14_CrossBiobank_Replication_Tests.tsv"), sep="\t", index=False)
r_all, p_all = pearsonr(df_rep["Z_Disc"], df_rep["Z_Rep_Aligned"])
r_abs, p_abs = pearsonr(df_rep["Z_Disc"].abs(), df_rep["Z_Rep_Aligned"].abs())
n_same = int(df_rep["Same_Sign"].sum())
n_nom = int(df_rep["Nom_Sig"].sum())
b_p = binomtest(n_same, len(df_rep), 0.5).pvalue

# Tekil benzersiz SNP bazinda replikasyon (bagimsizlik kontrolu)
df_rep_uniq = df_rep.sort_values("Z_Disc", key=lambda s: s.abs(), ascending=False).drop_duplicates(subset=["Lead_SNP"], keep="first")
ru_all, pu_all = pearsonr(df_rep_uniq["Z_Disc"], df_rep_uniq["Z_Rep_Aligned"])
ru_abs, pu_abs = pearsonr(df_rep_uniq["Z_Disc"].abs(), df_rep_uniq["Z_Rep_Aligned"].abs())
nu_same = int(df_rep_uniq["Same_Sign"].sum())
bu_p = binomtest(nu_same, len(df_rep_uniq), 0.5).pvalue

# Ayni hastalik ici kalibrasyon (90 lokus)
df_within = df_val[df_val["Pair_Tier"] == "Within_Disorder_Calibration"]
w_conc = (df_within["Direction_Aligned"] == "Concordant (+/+)").mean() * 100

print(f"\n  Within-Disorder Calibration (n={len(df_within)}): Yon uyumu = {w_conc:.1f}%")
print(f"  Tum Varyant-Kohort Tutarlilik Testleri (n={len(df_rep)}):")
print(f"    Signed Z Pearson r = {r_all:.4f} (P={p_all:.2e}) | |Z| Pearson r = {r_abs:.4f} (P={p_abs:.2e})")
print(f"    Yon Uyumu = {n_same}/{len(df_rep)} ({100*n_same/len(df_rep):.1f}%, binom P={b_p:.2e}) | Nominal P<0.05 = {n_nom} ({100*n_nom/len(df_rep):.1f}%)")
print(f"  Tekillestirilmis Benzersiz SNP Tutarliligi (n={len(df_rep_uniq)}):")
print(f"    Signed Z Pearson r = {ru_all:.4f} (P={pu_all:.2e}) | |Z| Pearson r = {ru_abs:.4f} (P={pu_abs:.2e})")
print(f"    Yon Uyumu = {nu_same}/{len(df_rep_uniq)} ({100*nu_same/len(df_rep_uniq):.1f}%, binom P={bu_p:.2e})")

# =============================================================================
# 2. TEMIZ ENRICHR (SADECE BIRINCIL 22, POZITIF BELIRTECLER, 'down' & DONOR YOK!)
# =============================================================================
print("\n[2/4] Temiz Enrichr analizi calistiriliyor (Sadece Birincil 22, 'down' ve GTEx donoru filtreleniyor)...", flush=True)

def query_enrichr(gene_list, libraries, desc):
    if len(gene_list) < 5:
        return []
    url_add = "https://maayanlab.cloud/Enrichr/addList"
    boundary = "----CEREBRAXBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"list\"\r\n\r\n" + "\n".join(gene_list) +
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"description\"\r\n\r\n{desc}" +
        f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(url_add, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            uid = json.loads(resp.read().decode("utf-8"))["userListId"]
    except Exception as e:
        print(f"    [Uyari] Enrichr addList basarisiz ({desc}): {e}")
        return []

    rows = []
    for lib in libraries:
        try:
            url_e = f"https://maayanlab.cloud/Enrichr/enrich?userListId={uid}&backgroundType={lib}"
            with urllib.request.urlopen(url_e, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get(lib, []):
                term, pval, zsc, csc, ov_genes, padj = item[1], float(item[2]), float(item[3]), float(item[4]), item[5], float(item[6])
                # KRITIK TEMIZLIK: 'down' soneki ve tekil GTEX donor barkodlarini KESINLIKLE AT!
                if re.search(r"\bdown\b", term, re.I) or "GTEX-" in term.upper():
                    continue
                if padj < 0.05 and len(ov_genes) >= 3:
                    rows.append({
                        "Gene_Set": desc, "Library": lib, "Term": term,
                        "P_Value": pval, "Adjusted_P": padj, "Combined_Score": round(csc, 2),
                        "N_Genes": len(ov_genes), "Genes": ", ".join(ov_genes)
                    })
        except Exception:
            pass
    return rows

clean_hgnc = lambda sub: sorted(set(sub["Resolved_Gene"].dropna().astype(str)) - {g for g in sub["Resolved_Gene"].dropna().astype(str) if g.startswith(("ENSG", "LOC_chr")) or g == "Unmapped"})

GENE_SETS = {
    "Primary22_All_Pleiotropic": clean_hgnc(df_prim),
    "Primary22_CrossDomain_NeuroPsych": clean_hgnc(df_prim[df_prim["Domain_Pair"] == "Cross_Neuro_Psych"]),
    "Primary22_IntraPsych": clean_hgnc(df_prim[df_prim["Domain_Pair"] == "Intra_Psych"]),
    "Primary22_CrossDomain_Antagonistic": clean_hgnc(df_prim[(df_prim["Domain_Pair"] == "Cross_Neuro_Psych") & (df_prim["Direction_Aligned"] == "Discordant (+/-)")]),
    "Primary22_CrossDomain_Agonistic": clean_hgnc(df_prim[(df_prim["Domain_Pair"] == "Cross_Neuro_Psych") & (df_prim["Direction_Aligned"] == "Concordant (+/+)")]),
}
for k, gl in GENE_SETS.items():
    print(f"    -> {k}: {len(gl)} benzersiz HGNC geni")

PATH_LIBS = ["GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human"]
CELL_LIBS = ["Allen_Brain_Atlas_10x_scRNA_2021", "PanglaoDB_Augmented_2021", "Descartes_Cell_Types_and_Tissue_2021", "GTEx_Tissues_V8_2023", "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X"]

all_enr = []
for gs_name, gl in GENE_SETS.items():
    all_enr.extend(query_enrichr(gl, PATH_LIBS + CELL_LIBS, gs_name))

df_enr = pd.DataFrame(all_enr)
if len(df_enr) > 0:
    df_enr = df_enr.sort_values(["Adjusted_P", "P_Value"])
    df_enr.to_csv(os.path.join(P14, "CEREBRAX_Phase14_Table4_Clean_Positive_Enrichments.tsv"), sep="\t", index=False)
    print(f"\n  Temiz pozitif zenginlestirme sonucu (Padj < 0.05, >=3 gen, 'down'/donor YOK): {len(df_enr)} terim")
    print("\n  --- TOP 10 YOLAK (GO/Reactome/KEGG) ---")
    p_sub = df_enr[df_enr["Library"].isin(PATH_LIBS)].head(10)
    print(p_sub[["Gene_Set", "Library", "Term", "P_Value", "Adjusted_P", "N_Genes"]].to_string(index=False))
    print("\n  --- TOP 12 TEK-HUCRE / DOKU / KROMATIN (SADECE POZITIF / UP / TF) ---")
    c_sub = df_enr[df_enr["Library"].isin(CELL_LIBS)].head(12)
    print(c_sub[["Gene_Set", "Library", "Term", "P_Value", "Adjusted_P", "N_Genes"]].to_string(index=False))
else:
    print("  [Uyari] Enrichr yaniti bos dondu.")

# =============================================================================
# 3. STANDART 10 Mb CLUMPING + GERCEK N ILE CIFT YONLU MR (TABLE 5)
# =============================================================================
print("\n[3/4] 10 Mb (10,000 kb) clumping + MHC/17q21.31 tekillestirme ile Cift Yonlu MR hesaplaniyor...", flush=True)

# Koordinat sozlugu
snp_pos = {}
with gzip.open(COORD_FILE, "rt", encoding="utf-8", errors="ignore") as f:
    f.readline()
    for line in f:
        p = line.strip().split()
        if len(p) >= 2 and ":" in p[0]:
            ch, bp = p[0].split(":")
            try:
                snp_pos[p[1]] = (ch.replace("chr", ""), int(bp))
            except ValueError:
                pass

MR_AXES = [
    ("ET_2024_Verma", "PGC_2026_GAD"),
    ("MS_2024_Science", "PGC_2024_BIP"),
    ("STROKE_2022_GIGA", "PGC_2025_MDD"),
    ("HEADACHE_Gen_Mbatchou", "PGC_2026_GAD"),
    ("MIG_2022_IHGC", "PGC_2025_MDD"),
    ("PD_2024_NatGen", "PGC_2025_MDD"),
]
mr_cohorts = sorted({c for pair in MR_AXES for c in pair})
mr_db = {}
for cname in mr_cohorts:
    sf = os.path.join(SUMSTATS_DIR, f"{cname}_Ready.sumstats.gz")
    d = {}
    with gzip.open(sf, "rt", encoding="utf-8", errors="ignore") as f:
        hdr = f.readline().strip().split()
        hm = {col.upper(): i for i, col in enumerate(hdr)}
        si, a1i, a2i, zi, ni = hm.get("SNP", 0), hm.get("A1", 1), hm.get("A2", 2), hm.get("Z", 3), hm.get("N", 4)
        for line in f:
            p = line.strip().split()
            if len(p) > zi:
                try:
                    zv = float(p[zi])
                    nv = float(p[ni]) if len(p) > ni else 100000.0
                    if not (math.isnan(zv) or math.isinf(zv)):
                        d[p[si]] = (p[a1i].upper(), p[a2i].upper(), zv, nv)
                except ValueError:
                    pass
    mr_db[cname] = d

def weighted_median_est(betas, weights, n_boot=500):
    order = np.argsort(betas)
    b_s, w_s = betas[order], weights[order]
    w_norm = w_s / np.sum(w_s)
    cum = np.cumsum(w_norm) - 0.5 * w_norm
    est = float(np.interp(0.5, cum, b_s))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(betas), size=len(betas))
        bs, ws = betas[idx], weights[idx]
        o = np.argsort(bs)
        wn = ws[o] / np.sum(ws[o])
        boots.append(float(np.interp(0.5, np.cumsum(wn) - 0.5 * wn, bs[o])))
    se = float(np.std(boots, ddof=1))
    p = float(2.0 * norm.sf(abs(est / se))) if se > 0 else 1.0
    return est, se, p

def run_mr_direction(exp_c, out_c, window_bp=10000000):
    d_exp, d_out = mr_db[exp_c], mr_db[out_c]
    # P < 5e-8 (|Z| >= 5.45131) enstruman adaylari
    cands = []
    for rs, (a1_1, a2_1, z1, n1) in d_exp.items():
        if abs(z1) >= 5.45131 and rs in d_out and rs in snp_pos:
            a1_2, a2_2, z2, n2 = d_out[rs]
            z2a = align_single(a1_1, a2_1, a1_2, a2_2, z2)
            if z2a is not None:
                ch, bp = snp_pos[rs]
                cands.append((abs(z1), rs, ch, bp, z1, n1, z2a, n2))
    cands.sort(reverse=True, key=lambda x: x[0])

    # 10 Mb clumping + MHC / 17q21.31 tek enstruman kurali
    kept = []
    used_pos = []
    used_super = set()
    for _, rs, ch, bp, z1, n1, z2a, n2 in cands:
        sup = "MHC" if (ch == "6" and 25000000 <= bp <= 34000000) else ("17q21" if (ch == "17" and 43500000 <= bp <= 47000000) else None)
        if sup and sup in used_super:
            continue
        if any(uc == ch and abs(bp - ubp) < window_bp for uc, ubp in used_pos):
            continue
        kept.append((rs, z1, n1, z2a, n2))
        used_pos.append((ch, bp))
        if sup:
            used_super.add(sup)

    n_iv = len(kept)
    if n_iv < 2:
        return None
    z_x = np.array([x[1] for x in kept], dtype=np.float64)
    n_x = np.array([x[2] for x in kept], dtype=np.float64)
    z_y = np.array([x[3] for x in kept], dtype=np.float64)
    n_y = np.array([x[4] for x in kept], dtype=np.float64)

    # Standartlastirilmis etki: 2*MAF*(1-MAF) Wald oraninda sadeleşir!
    sgn = np.where(z_x > 0, 1.0, -1.0)
    bx = np.abs(z_x) / np.sqrt(n_x)
    by = (z_y * sgn) / np.sqrt(n_y)
    se_y = 1.0 / np.sqrt(n_y)
    se_ratio = se_y / bx
    wald = by / bx
    w = 1.0 / (se_ratio ** 2)

    mean_f = float(np.mean(z_x ** 2))
    # Multiplicative random-effects IVW
    ivw_b = float(np.sum(w * wald) / np.sum(w))
    q_j = w * ((wald - ivw_b) ** 2)
    Q = float(np.sum(q_j))
    df_q = n_iv - 1
    q_p = float(chi2.sf(Q, df_q))
    i2 = max(0.0, 100.0 * (Q - df_q) / Q) if Q > 0 else 0.0
    phi = max(1.0, Q / df_q)
    ivw_se = float(math.sqrt(phi / np.sum(w)))
    ivw_p = float(2.0 * norm.sf(abs(ivw_b / ivw_se)))

    # Radial outlier-corrected IVW (Qj > 3.8415)
    inliers = q_j <= 3.8415
    n_out = int((~inliers).sum())
    if inliers.sum() >= 2 and n_out > 0:
        wi, waldi = w[inliers], wald[inliers]
        oc_b = float(np.sum(wi * waldi) / np.sum(wi))
        Q_in = float(np.sum(wi * ((waldi - oc_b) ** 2)))
        phi_in = max(1.0, Q_in / (inliers.sum() - 1))
        oc_se = float(math.sqrt(phi_in / np.sum(wi)))
        oc_p = float(2.0 * norm.sf(abs(oc_b / oc_se)))
    else:
        oc_b, oc_se, oc_p = ivw_b, ivw_se, ivw_p

    # Weighted Median
    wm_b, wm_se, wm_p = weighted_median_est(wald, w) if n_iv >= 3 else (float("nan"), float("nan"), float("nan"))

    # MR-Egger (sadece N_IV >= 5 ise guvenilir)
    if n_iv >= 5:
        W_e = 1.0 / (se_y ** 2)
        X = np.column_stack([np.ones(n_iv), bx])
        XtWX = X.T @ (W_e[:, None] * X)
        cov = np.linalg.inv(XtWX)
        coef = cov @ (X.T @ (W_e * by))
        resid = by - X @ coef
        s2 = max(1.0, float(np.sum(W_e * (resid ** 2)) / (n_iv - 2)))
        se_int = math.sqrt(s2 * cov[0, 0])
        egger_int_p = float(2.0 * t_dist.sf(abs(coef[0] / se_int), df=n_iv - 2))
    else:
        egger_int_p = float("nan")

    # MR-Steiger (gozlenen olcek R2 = sum Z^2 / (Z^2 + N - 2))
    r2_exp = float(np.sum((z_x ** 2) / (z_x ** 2 + n_x - 2.0)))
    r2_out = float(np.sum((z_y ** 2) / (z_y ** 2 + n_y - 2.0)))
    r_e, r_o = math.sqrt(min(0.9999, r2_exp)), math.sqrt(min(0.9999, r2_out))
    st_z = (math.atanh(r_e) - math.atanh(r_o)) / math.sqrt(1.0 / (np.median(n_x) - 3.0) + 1.0 / (np.median(n_y) - 3.0))
    st_p = float(norm.sf(st_z))
    st_p_str = "< 1.00e-300" if st_p < 1e-300 else f"{st_p:.2e}"
    gcov_i, _ = gcov_map.get(tuple(sorted([exp_c, out_c])), (float("nan"), float("nan")))

    return {
        "Exposure": exp_c, "Outcome": out_c, "N_IVs_10Mb": n_iv, "Mean_F": round(mean_f, 1),
        "IVW_Beta": round(ivw_b, 4), "IVW_SE": round(ivw_se, 4), "IVW_P": round(ivw_p, 5),
        "Cochran_Q": round(Q, 1), "I2_Pct": round(i2, 1), "Outliers_Rem": n_out,
        "OutlierCorr_Beta": round(oc_b, 4), "OutlierCorr_SE": round(oc_se, 4), "OutlierCorr_P": round(oc_p, 5),
        "WM_Beta": round(wm_b, 4) if not math.isnan(wm_b) else np.nan,
        "WM_P": round(wm_p, 5) if not math.isnan(wm_p) else np.nan,
        "Egger_Int_P": round(egger_int_p, 4) if not math.isnan(egger_int_p) else np.nan,
        "R2_Exp_Pct": round(100 * r2_exp, 3), "R2_Out_Pct": round(100 * r2_out, 3),
        "Steiger_P": st_p_str, "LDSC_gcov_int": round(gcov_i, 4) if not math.isnan(gcov_i) else np.nan
    }

mr_rows = []
for a, b in MR_AXES:
    for e, o in [(a, b), (b, a)]:
        res = run_mr_direction(e, o, window_bp=10000000)
        if res:
            mr_rows.append(res)
df_mr = pd.DataFrame(mr_rows)
df_mr.to_csv(os.path.join(P14, "CEREBRAX_Phase14_Table5_MR_10Mb_Clean.tsv"), sep="\t", index=False)
print(df_mr[["Exposure", "Outcome", "N_IVs_10Mb", "Mean_F", "IVW_Beta", "IVW_SE", "IVW_P", "I2_Pct", "OutlierCorr_Beta", "OutlierCorr_P", "WM_P", "Egger_Int_P", "Steiger_P"]].to_string(index=False))

# =============================================================================
# 4. HATASIZ YENI FIGURE 2 VE FIGURE 3 CIZIMI (300 DPI PDF + PNG)
# =============================================================================
print("\n[4/4] Hatasiz Figure 2 ve Figure 3 uretiliyor...", flush=True)
plt.rcParams.update({"font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})

# --- FIGURE 2 (Phase 14A kesin rakamlariyla) ---
fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.3), dpi=300)

# Panel A: Marginal + Intersection Validation Funnel (2155 -> 1247 / 943 -> 706 -> 436 Primary)
ax = axes[0]
n_cand = len(df_all_cand)
n_coloc = int((df_all_cand["COLOC_PP_H4_Real"] >= 0.75).sum())
n_placo = int((df_all_cand["P_PLACO_Sym"] < 5e-8).sum())
n_both = len(df_val)
n_prim = len(df_prim)
stages = ["Consolidated\nCandidates", "Multi-SNP COLOC\n(PP.H4 ≥ 0.75)*", "Symmetric PLACO\n(P < 5×10⁻⁸)*", "Jointly Colocalized\n(All 861 Pairs)", "Primary 22\nNon-Redundant"]
vals = [n_cand, n_coloc, n_placo, n_both, n_prim]
cols = ["#94a3b8", "#60a5fa", "#3b82f6", "#1d4ed8", "#0f766e"]
bars = ax.bar(range(5), vals, color=cols, width=0.6, edgecolor="#1e293b", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 35, f"{v:,}", ha="center", va="bottom", fontweight="bold", fontsize=8.5)
ax.set_xticks(range(5))
ax.set_xticklabels(stages, fontsize=7.6, rotation=16, ha="right")
ax.set_ylabel("Number of Independent Loci", fontweight="bold")
ax.set_ylim(0, max(vals) * 1.16)
ax.text(0.98, 0.94, "*Marginal pass counts\n(intersection = 706)", transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color="#334155")
ax.set_title("A. Local Pleiotropy Filtering & Tier Partition", fontweight="bold", loc="left", pad=10)

# Panel B: Top 15 Primary 22 Non-Redundant Hub Genes
ax = axes[1]
top15 = df_hubs.head(15).iloc[::-1]
y = np.arange(len(top15))
ax.barh(y, top15["Unique_Primary_Pairs"], color="#2563eb", alpha=0.85, label="Total Primary Pairs")
ax.barh(y, top15["Cross_Domain_Pairs"], color="#dc2626", alpha=0.88, label="Cross-Domain (Neuro×Psych)")
for yi, tot, cr in zip(y, top15["Unique_Primary_Pairs"], top15["Cross_Domain_Pairs"]):
    ax.text(tot + 0.18, yi, f"{tot} ({cr})" if cr > 0 else f"{tot}", va="center", ha="left", fontsize=7.8, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(top15["Gene"], fontsize=8.0, fontstyle="italic")
ax.set_xlabel("Unique Primary Disorder Pairs (Cross-Domain)", fontweight="bold")
ax.set_xlim(0, top15["Unique_Primary_Pairs"].max() + 2.4)
ax.legend(loc="lower right", frameon=True, fontsize=7.5)
ax.set_title("B. Top 15 Non-Redundant Hub Genes (Primary 22)", fontweight="bold", loc="left", pad=10)

# Panel C: Primary 22 Allele-Aligned Effect Directionality (n=436)
ax = axes[2]
z1 = np.clip(df_prim["Z_Trait1"].values, -15, 15)
z2 = np.clip(df_prim["Z_Trait2_Aligned"].values, -15, 15)
is_conc = (df_prim["Direction_Aligned"] == "Concordant (+/+)").values
nc, nd = int(is_conc.sum()), int((~is_conc).sum())
cross_p = df_prim[df_prim["Domain_Pair"] == "Cross_Neuro_Psych"]
cc = int((cross_p["Direction_Aligned"] == "Concordant (+/+)").sum())
cd = int((cross_p["Direction_Aligned"] == "Discordant (+/-)").sum())
ax.axhline(0, color="#64748b", ls="--", lw=0.8, alpha=0.7)
ax.axvline(0, color="#64748b", ls="--", lw=0.8, alpha=0.7)
ax.scatter(z1[is_conc], z2[is_conc], s=20, alpha=0.65, color="#0d9488", edgecolors="none", label=f"Concordant (+/+): {nc} ({100*nc/len(df_prim):.1f}%)")
ax.scatter(z1[~is_conc], z2[~is_conc], s=24, alpha=0.82, color="#dc2626", edgecolors="none", label=f"Discordant (+/-): {nd} ({100*nd/len(df_prim):.1f}%)")
ax.set_xlabel("Trait 1 Signed Z-score (clipped at ±15)", fontweight="bold")
ax.set_ylabel("Trait 2 Allele-Aligned Z-score (clipped at ±15)", fontweight="bold")
ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)
ax.legend(loc="upper left", frameon=True, fontsize=7.6)
ax.text(0.97, 0.04, f"Cross-Domain (n={len(cross_p)}):\n{100*cc/len(cross_p):.1f}% (+/+) vs. {100*cd/len(cross_p):.1f}% (+/-)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"))
ax.set_title(f"C. Primary 22 Allele-Aligned Directionality (n={len(df_prim)})", fontweight="bold", loc="left", pad=10)

plt.tight_layout(w_pad=2.2)
plt.savefig(os.path.join(P14, "Figure2_Phase14_Pleiotropy_Hubs_Directionality.pdf"))
plt.savefig(os.path.join(P14, "Figure2_Phase14_Pleiotropy_Hubs_Directionality.png"), dpi=300)
plt.close()

# --- FIGURE 3 (Temiz Pozitif Zenginlestirmeler + 10 Mb MR Forest Plot — Kırpılma YOK!) ---
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})

# Panel A: Clean Positive-Only Enrichments
ax = axes[0]
if len(df_enr) > 0:
    # En anlamli benzersiz 10 terimi sec (yolak + hucre/TF dengeli)
    top_e = df_enr.drop_duplicates(subset=["Term"]).head(10).iloc[::-1]
    labels_e = []
    for _, r in top_e.iterrows():
        short_gs = r["Gene_Set"].replace("Primary22_", "").replace("_", " ")
        t_clean = re.sub(r"\s*\((GO|R-HSA):[^)]+\)", "", str(r["Term"]))
        if len(t_clean) > 42: t_clean = t_clean[:39] + "..."
        labels_e.append(f"{t_clean}\n[{short_gs}]")
    y_e = np.arange(len(top_e))
    neg_log = -np.log10(np.maximum(top_e["Adjusted_P"].values, 1e-30))
    colors_e = ["#0f766e" if lib in CELL_LIBS else "#2563eb" for lib in top_e["Library"]]
    ax.barh(y_e, neg_log, color=colors_e, alpha=0.88, edgecolor="#1e293b", linewidth=0.5)
    ax.axvline(-math.log10(0.05), color="#dc2626", ls="--", lw=1.0, label="FDR = 0.05")
    ax.set_yticks(y_e)
    ax.set_yticklabels(labels_e, fontsize=7.6)
    ax.set_xlabel("-log10(Adjusted P-value, BH-FDR)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
ax.set_title("A. Positive Pathway, Cell-Type & TF Enrichments (Primary 22)", fontweight="bold", loc="left", pad=10)

# Panel B: 10 Mb Clumped MR Forest Plot
ax = axes[1]
SHORT_MR = {
    "ET_2024_Verma": "Essential Tremor", "PGC_2026_GAD": "Anxiety (GAD)",
    "MS_2024_Science": "Multiple Sclerosis", "PGC_2024_BIP": "Bipolar Disorder",
    "STROKE_2022_GIGA": "Ischemic Stroke", "PGC_2025_MDD": "Major Depression",
    "HEADACHE_Gen_Mbatchou": "Broad Headache", "MIG_2022_IHGC": "Migraine",
    "PD_2024_NatGen": "Parkinson's Disease"
}
df_mr_plot = df_mr.iloc[::-1].reset_index(drop=True)
y_m = np.arange(len(df_mr_plot))
mr_lbls = [f"{SHORT_MR.get(r['Exposure'], r['Exposure'])} → {SHORT_MR.get(r['Outcome'], r['Outcome'])} (n={r['N_IVs_10Mb']})" for _, r in df_mr_plot.iterrows()]
betas = df_mr_plot["IVW_Beta"].values
ci95 = 1.96 * df_mr_plot["IVW_SE"].values
ax.axvline(0, color="#dc2626", ls="--", lw=1.0)
ax.errorbar(betas, y_m, xerr=ci95, fmt="o", color="#1e40af", ecolor="#64748b", elinewidth=1.4, capsize=3.5, markersize=5.5)
ax.set_yticks(y_m)
ax.set_yticklabels(mr_lbls, fontsize=8.0)
ax.set_xlabel("Standardized IVW Causal Estimate (Beta ± 95% CI)", fontweight="bold")
ax.set_title("B. Bidirectional MR Across Six Clinical Axes (10 Mb Clumping)", fontweight="bold", loc="left", pad=10)

plt.tight_layout(w_pad=2.5)
plt.savefig(os.path.join(P14, "Figure3_Phase14_CleanEnrichments_and_MR10Mb.pdf"))
plt.savefig(os.path.join(P14, "Figure3_Phase14_CleanEnrichments_and_MR10Mb.png"), dpi=300)
plt.close()

print(f"\nPHASE 14B TAMAMLANDI ({time.time()-t0:.1f}s) -> {P14}")
