import sys
import allel
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt

def filter_samples_with_min_snps(genotypes, samples, min_snps=3):
    """
    Remove samples (columns) that have fewer than `min_snps` called genotypes.
    
    Returns
    --------
    filtered_genotypes: allel.GenotypeArray with removed columns
    kept_samples: list of sample names kept
    kept_indices: list of original indices kept
    """
    called_per_sample = genotypes.is_called().sum(axis=0)
    keep_mask = called_per_sample >= min_snps
    print(f"Samples kept: {keep_mask.sum()} / {len(keep_mask)} "
          f"({keep_mask.sum()/len(keep_mask)*100:.2f}%)")
    print("Removed samples:", [samples[i] for i in range(len(samples)) if not keep_mask[i]])
    filtered_genotypes = genotypes[:, keep_mask]
    kept_samples = [samples[i] for i in range(len(samples)) if keep_mask[i]]
    kept_indices = np.where(keep_mask)[0].tolist()
    return filtered_genotypes, kept_samples, kept_indices

def filter_by_maf(genotypes, min_maf=0.05):
    """
    Remove SNPs with a Minor Allele Frequency (MAF) under the given threshold.
    
    Returns
    --------
    allel.GenotypeArray with removed columns
    """
    ac = genotypes.count_alleles()
    af = ac.to_frequencies()
    # MAF = fréquence de l'allèle le moins fréquent
    maf = np.min(af[:, :2], axis=1)  # prend les 2 premiers allèles
    mask = maf >= min_maf
    print(f"MAF ≥ {min_maf}: kept {mask.sum()} / {len(mask)} variants "
          f"({mask.sum()/len(mask)*100:.2f}%).")
    return genotypes[mask]

def truly_fixed(geno, idx):
    n_alt = geno[:, idx].to_n_alt()
    out = np.zeros(geno.shape[0], dtype=bool)
    for i, row in enumerate(n_alt):
        vals = row[~np.isnan(row)]
        if len(vals) == 0:
            out[i] = False
        else:
            # all 0 or all 2
            out[i] = (np.all(vals == 0) or np.all(vals == 2))
    return out

def heterozygous(geno, idx):
    """
    True heterozygosity: at least one 0/1 genotype.
    """
    n_alt = geno[:, idx].to_n_alt()
    return np.any(n_alt == 1, axis=1)

def fixed_contrasts(geno, idx):
    """
    Returns a boolean array where True means:
    - at least one 0/0 genotype
    - at least one 1/1 genotype
    - no 0/1 genotypes
    """
    n_alt = geno[:, idx].to_n_alt()
    has_ref = np.any(n_alt == 0, axis=1)
    has_alt = np.any(n_alt == 2, axis=1)
    has_het = np.any(n_alt == 1, axis=1)
    return has_ref & has_alt & ~has_het

def polymorphic(geno, idx):
    """
    Returns a boolean array where True means:
    - at least one REF allele is present in the group
      (either heterozygous 0/1 or homozygous ALT 1/1)
    """
    n_alt = geno[:, idx].to_n_alt()
    has_ref = np.any((n_alt == 0) | (n_alt == 1), axis=1)
    has_alt = np.any((n_alt == 1) | (n_alt == 2), axis=1)
    return has_ref & has_alt

def compute_stats_clean(geno, anc_idx, mod_idx):
  '''
  Description
      Compute comparative SNP statistics between two groups (Ancient vs Modern).
      The function classifies SNPs as fixed or polymorphic within each group,
      identifies shared vs group‑specific polymorphism, computes allele frequencies
      per group, and quantifies directional differences in allele frequency between
      Ancient and Modern for SNPs polymorphic in both groups.
  Input
      geno     : scikit‑allel GenotypeArray
      anc_idx  : list of indices for Ancient samples
      mod_idx  : list of indices for Modern samples
  Output
      Dictionary containing per‑SNP comparative statistics:
          - total_snps                     : SNPs callable in both groups
          - ancient_poly_only              : polymorphic only in Ancient
          - modern_poly_only               : polymorphic only in Modern
          - both_poly                      : polymorphic in both groups
          - both_fix                       : fixed in both groups
          - both_poly_ancient_more         : shared polymorphic SNPs where Ancient AF > Modern AF
          - both_poly_modern_more          : shared polymorphic SNPs where Modern AF > Ancient AF
          - both_poly_ancient_much_more    : Ancient AF ≥ 2× Modern AF
          - both_poly_modern_much_more     : Modern AF ≥ 2× Ancient AF
          - unclassified                   : callable SNPs not fitting any category above
  '''
    anc_fix = truly_fixed(geno, anc_idx)
    mod_fix = truly_fixed(geno, mod_idx)
    anc_poly = polymorphic(geno, anc_idx)
    mod_poly = polymorphic(geno, mod_idx)
    # Valid SNPs = at least one called genotype in both groups
    anc_called = ~np.isnan(geno[:, anc_idx].to_n_alt()).all(axis=1)
    mod_called = ~np.isnan(geno[:, mod_idx].to_n_alt()).all(axis=1)
    valid = anc_called & mod_called
    unclassified = valid & ~(anc_poly | mod_poly | anc_fix | mod_fix)
    shared_poly = anc_poly & mod_poly & valid
    # ALT counts
    n_alt = geno.to_n_alt()
    anc_alt_count = np.nansum(n_alt[:, anc_idx] == 1, axis=1) + 2 * np.nansum(n_alt[:, anc_idx] == 2, axis=1)
    mod_alt_count = np.nansum(n_alt[:, mod_idx] == 1, axis=1) + 2 * np.nansum(n_alt[:, mod_idx] == 2, axis=1)
    # Callable samples per SNP
    anc_called_per_snp = np.sum(~np.isnan(n_alt[:, anc_idx]), axis=1)
    mod_called_per_snp = np.sum(~np.isnan(n_alt[:, mod_idx]), axis=1)
    # Allele frequencies
    anc_alt_freq = np.divide( anc_alt_count, 2 * anc_called_per_snp, out=np.zeros_like(anc_alt_count, dtype=float), where=anc_called_per_snp > 0 )
    mod_alt_freq = np.divide( mod_alt_count, 2 * mod_called_per_snp, out=np.zeros_like(mod_alt_count, dtype=float), where=mod_called_per_snp > 0 )
    return {
        "total_snps": valid.sum(),
        "ancient_poly_only": np.sum(anc_poly & ~mod_poly & valid),
        "modern_poly_only": np.sum(mod_poly & ~anc_poly & valid),
        "both_poly": np.sum(anc_poly & mod_poly & valid),
        "both_fix": np.sum(anc_fix & mod_fix & valid),
        "both_poly_ancient_more": np.sum((anc_alt_freq > mod_alt_freq) & shared_poly),
        "both_poly_modern_more":np.sum((mod_alt_freq > anc_alt_freq) & shared_poly),
        "both_poly_ancient_much_more": np.sum((anc_alt_freq >= 2 * mod_alt_freq) & shared_poly),
        "both_poly_modern_much_more": np.sum((mod_alt_freq >= 2 * anc_alt_freq) & shared_poly),
        "unclassified": unclassified.sum()
    }

def snp_count_ancient_modern_shared(vcf,pop,out,maf=0.05):
    '''
  Description
      Compute SNP statistics for two groups (Ancient vs Modern) within the same VCF.
  Input
      vcf : txt; path to VCF file
      pop : tsv; sample + population (here, "Ancient" or "Modern")
      out : txt; output path for the summary CSV
      maf : float; minimum minor allele frequency for filtering
  Output
      - Printed statistics (raw + MAF‑filtered) for Ancient and Modern groups
      - CSV summary table containing group‑wise SNP counts
  '''
    print("Read VCF")
    callset = allel.read_vcf(vcf)
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    smpl = callset['samples']
    genotypes, smpl, kept_indices = filter_samples_with_min_snps(genotypes, smpl, min_snps=3) # Remove samples with < 3 SNPs
    print("Read POP")
    pop_df = pd.read_csv(pop, sep="\t", header=None, names=["Sample", "Population"])
    pop_df = pop_df[pop_df["Sample"].isin(smpl)]        # Match samples directly
    # Identify ancient vs modern
    anc_samples = pop_df[pop_df["Population"] == "Ancient"]["Sample"].tolist()
    mod_samples = pop_df[pop_df["Population"] == "Modern"]["Sample"].tolist()
    anc_idx = [list(smpl).index(s) for s in anc_samples]
    mod_idx = [list(smpl).index(s) for s in mod_samples]
    print(f"Ancient samples: {len(anc_idx)}")
    print(f"Modern samples: {len(mod_idx)}")
    print("\n=== Raw VCF statistics ===")
    # BEFORE MAF filtering
    stats_raw = compute_stats_clean(genotypes, anc_idx, mod_idx)
    for k, v in stats_raw.items(): print(f"{k}: {v}")
    # AFTER MAF filtering
    genotypes_maf = filter_by_maf(genotypes, min_maf=maf)
    stats_maf = compute_stats_clean(genotypes_maf, anc_idx, mod_idx)
    for k, v in stats_maf.items(): print(f"{k}: {v}")
    # EXPORT Summary table
    summary_df = pd.DataFrame([
    {"Filter": "Raw", **stats_raw},
    {"Filter": "MAF ≥ 0.05", **stats_maf}
    ])
    summary_df.to_csv(out, index=False)
    print(f"\nSummary table exported to:\n{out}")
    print(summary_df)

def snp_count_snps_one_group_one_vcf(vcf,maf,min=3):
    '''
  Description
      Compute SNP category counts for a single group of samples in one VCF.
      The function loads a VCF, filters samples with too few SNPs, then counts
      SNP categories (heterozygous, fixed, fixed contrasts, polymorphic) before
      and after MAF filtering.
  Input
      vcf : txt; path to VCF file
      maf : float; minimum minor allele frequency for filtering
      min : int; minimum number of SNPs required per sample
  Output
      - Printed SNP counts before and after MAF filtering:
          * total SNPs
          * heterozygous SNPs
          * fixed SNPs
          * fixed contrasts (0/0 vs 1/1, no heterozygotes)
          * polymorphic SNPs
  '''
    print("Read VCF")
    callset = allel.read_vcf(vcf)
    smpl = callset["samples"]
    genotypes = allel.GenotypeArray(callset["calldata/GT"])
    genotypes, smpl, kept_indices = filter_samples_with_min_snps(genotypes, smpl, min_snps=min) # Remove samples with < min SNPs
    all_idx = list(range(genotypes.shape[1]))
    # no MAF
    n_before = genotypes.shape[0]
    het_before = np.sum(heterozygous(genotypes, all_idx))
    fix_before = np.sum(truly_fixed(genotypes, all_idx))
    contrast_before = np.sum(fixed_contrasts(genotypes, all_idx))
    poly_before = np.sum(polymorphic(genotypes, all_idx))
    print("\n=== BEFORE MAF FILTERING ===")
    print(f"Total SNPs: {n_before}")
    print(f"Heterozygous SNPs: {het_before}")
    print(f"Fixed SNPs: {fix_before}")
    print(f"Fixed contrasts (0/0 & 1/1, no 0/1): {contrast_before}")
    print(f"Polymorphic SNPs: {poly_before}")
    # MAF
    genotypes_maf = filter_by_maf(genotypes, min_maf=maf)
    n_after = genotypes_maf.shape[0]
    het_after = np.sum(heterozygous(genotypes_maf, all_idx))
    fix_after = np.sum(truly_fixed(genotypes_maf, all_idx))
    contrast_after = np.sum(fixed_contrasts(genotypes_maf, all_idx))
    poly_after = np.sum(polymorphic(genotypes_maf, all_idx))
    print("\n=== AFTER MAF FILTERING ===")
    print(f"Total SNPs: {n_after}")
    print(f"Heterozygous SNPs: {het_after}")
    print(f"Fixed SNPs: {fix_after}")
    print(f"Fixed contrasts (0/0 & 1/1, no 0/1): {contrast_after}")
    print(f"Polymorphic SNPs: {poly_after}")

# --------------------------------------------------------- 
# Parameters
# ---------------------------------------------------------

input_path = r"" # vcf.gz
poput_path = r"" # tsv, 2 columns: samples + population,
output_path = r"" # csv, output path
minimum_maf = 0.05  # change as needed
snp_count_ancient_modern_shared(vcf=input_path,pop=poput_path,out=output_path,maf=minimum_maf)
snp_count_snps_one_group_one_vcf(vcf=input_path,maf=minimum_maf)
