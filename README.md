# CEREBRAX

**Cross-Disorder Neuropsychiatric Genomics, Local Pleiotropy, and Mendelian Randomization Pipeline**

Developed by **Dr. Cihat Özgüncü** (Department of Neurology, Selçuk University Faculty of Medicine).

## Overview
This repository contains the core Python analysis scripts implementing the CEREBRAX Phase 14 pipeline for the 22-disorder Primary Non-Redundant Discovery Tier and 20-cohort Cross-Biobank Consistency / Within-Disorder Calibration Tier:

- **Pairwise Allele Alignment & Symmetric ZCA-PLACO:** Empirical null-correlation estimation and zero-phase component analysis whitening across autosomal non-palindromic HapMap Phase 3 variants.
- **Regional Multi-SNP Colocalization & Conditional Fine-Mapping:** Continuous Wakefield Approximate Bayes Factor (ABF) colocalization (±250 kb windows) and 95% Bayesian credible set (`CS95`) resolution.
- **Bidirectional Two-Sample Mendelian Randomization:** 10 Mb clumping with Extended MHC (`chr6:25–34 Mb`) and `chr17q21.31` (`chr17:43.5–47.0 Mb`) consolidation, random-effects IVW, radial outlier correction, Weighted Median, MR-Egger intercept, and observed-scale MR-Steiger directionality testing.
- **Functional & Enrichment Auditing:** Primary-tier deduplicated Enrichr querying (removing downregulated terms and single-donor GTEx barcodes) and publication figure generation.

## Software Environment
- Python 3.12.3 (`NumPy 1.26.4`, `SciPy 1.11.4`, `Pandas 2.1.4`, `Matplotlib 3.6.3`)
- Legacy `LDSC v1.0.1` (Python 2.7 environment)

## Data Availability
No individual-level or consortium summary-statistics files are redistributed in this repository. All GWAS and WGS summary statistics analyzed in the study are publicly available through their respective consortia (PGC, FinnGen Release 11, UK Biobank, GIGASTROKE, IHGC, IMSGC, IPDGC, ILAE, Project MinE, and MVP) as detailed in Supplementary Table S1 of the manuscript.

## Citation
If you use or adapt scripts from this repository, please cite the accompanying CEREBRAX manuscript (citation details will be updated upon publication).
