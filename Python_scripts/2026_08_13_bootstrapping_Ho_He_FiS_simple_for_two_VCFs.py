import sys
import allel
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
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

def unified_ho_he_fis_per_pop(genotypes, subpops, populations, samples):
    pop_results = []
    ind_results = []
    for pop, indices in zip(populations, subpops):
        # Subset genotypes for this population
        geno_sub = genotypes[:, indices]
        # --- Expected heterozygosity (He) ---
        ac = geno_sub.count_alleles()
        af = ac.to_frequencies()
        n_chrom = ac.sum(axis=1).astype(float) # total number of called chromosomes per locus
        he_locus = allel.heterozygosity_expected(af, ploidy=2).astype(float)
        mask_no_data = (n_chrom == 0)
        he_locus[mask_no_data] = np.nan
        valid_n = (n_chrom > 1) # loci with at least 2 gene copies (n > 1)
        he_unb_locus = np.full_like(he_locus, np.nan, dtype=float)
        he_locus[ac.sum(axis=1) == 0] = np.nan # initialize
        he_unb_locus[valid_n] = (n_chrom[valid_n] / (n_chrom[valid_n] - 1.0)) * he_locus[valid_n] # apply correction
        he_pop = np.nanmean(he_unb_locus)
        # --- Observed heterozygosity (Ho) ---
        het = geno_sub.is_het()
        called = geno_sub.is_called()
        ho_individuals = []
        for i, idx in enumerate(indices):
            n_called_i = np.sum(called[:, i])
            if n_called_i > 0:
                ho_i = np.sum(het[:, i] & called[:, i]) / n_called_i
            else:
                ho_i = np.nan
            ho_individuals.append(ho_i)
            ind_results.append({ "Sample": samples[idx], "Population": pop, "Ho": ho_i })
        ho_pop = np.nanmean(ho_individuals)
        # --- Wright's FIS ---
        if he_pop > 0 and np.isfinite(he_pop):
            fis_pop = (he_pop - ho_pop) / he_pop
        else:
            fis_pop = np.nan
        pop_results.append({ "Population": pop, "Ho": ho_pop, "He": he_pop, "FIS": fis_pop, "N": len(indices)})
    return pd.DataFrame(pop_results), pd.DataFrame(ind_results)

def run_wilcoxon_test(ho_df):
    '''
    Only for calibration (DART vs WGS for example)
    '''
    ho_df = ho_df.copy()
    ho_df["Sample_clean"] = ho_df["Sample"].str.replace("^DART-", "", regex=True)
    paired = ho_df.pivot(index="Sample_clean", columns="Population", values="Ho")
    paired = paired.dropna()
    if paired.shape[1] != 2 or paired.shape[0] == 0:
        print("Wilcoxon test skipped: need exactly 2 matched groups with paired samples.")
        return None
    groups = paired.columns
    a = paired[groups[0]].values
    b = paired[groups[1]].values
    stat, p = stats.wilcoxon(a, b)
    print(f"Wilcoxon signed-rank W={stat:.3f}, p={p:.4f}")
    return {"W": stat, "p_value": p, "group1": groups[0], "group2": groups[1], "n_pairs": len(a)}

def run_dunn_test(ho_df):
    groups = [group["Ho"].dropna().values for _, group in ho_df.groupby("Population")]
    # Run Kruskal-Wallis first
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H={stat:.3f}, p={p:.4f}")
    if p < 0.05:
        print("Significant differences detected. Running Dunn test...")
        dunn = sp.posthoc_dunn(ho_df, val_col="Ho", group_col="Population", p_adjust="bonferroni")
        return dunn
    else:
        print("No significant differences between populations.")
        return None

def group_samples_id_by_pop(samples, sample_to_pop, populations, pop_df, smpl):
    try:
        subpops = [[i for i, s in enumerate(samples) if sample_to_pop[s] == pop] for pop in populations]
        subpops = [
            [i for i, s in enumerate(samples) if s in sample_to_pop and sample_to_pop[s] == pop]
            for pop in sorted(pop_df["Population"].unique())
        ]
    except:
        missing = [s for s in smpl if s not in pop_df["Sample"].tolist()]
        print("Samples in VCF but NOT in pop_df:", missing)
        extra = [s for s in pop_df["Sample"].tolist() if s not in smpl]
        print("Samples in pop_df but NOT in VCF:", extra)
        sys.exit()
    return subpops

def delete_small_groups(subpops, populations, N=2):
    valid = [i for i, inds in enumerate(subpops) if len(inds) >= N]
    subpops = [subpops[i] for i in valid]
    populations = [populations[i] for i in valid]
    return subpops, populations

def subsample_snps(genotypes, n_target, seed=None):
    """
    Randomly subsample a GenotypeArray (or similar array-like) down to n_target SNPs.
    """
    rng = np.random.default_rng(seed)
    n_snps = genotypes.shape[0]
    if n_target > n_snps:
        raise ValueError(f"n_target ({n_target}) > available SNPs ({n_snps})")
    idx = rng.choice(n_snps, size=n_target, replace=False)
    idx.sort()  # restore the genomic order
    return genotypes.take(idx, axis=0), idx

callset1=allel.read_vcf(r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\salmo_trutta_modern_mpileup_bisnps_refilled_Q30_DP5.vcf.gz")
callset2=allel.read_vcf(r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\salmo_trutta_ancient_mpileup_bisnps_refilled_Q30_DP5.vcf.gz")
pop_df1_full = pd.read_csv(r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\modernes_3_lineagess.txt", sep="\t", header=None, names=["Sample", "Population"])
pop_df2_full = pd.read_csv(r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\anciens_1pop.txt", sep="\t", header=None, names=["Sample", "Population"])
save_dunn = r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\Dunn_trutta_lineages_MAF.csv"
save_stats= r"c:\Users\pajacques\Documents\2025-07-09_moderne_mapping\AN_Trutta_separation\Ho_stats_trutta_lineages_subsampled_MAF.csv"
n_iter = 100
all_wilcoxon_results = []
all_dunn_results = []
all_stats = []
all_pop_inclusion = []
# Get GenotypeArrays
genotypes1 = allel.GenotypeArray(callset1['calldata/GT'])
genotypes2 = allel.GenotypeArray(callset2['calldata/GT'])
# filters Genotypes by MAF
genotypes1 = filter_by_maf(genotypes1, min_maf=0.05)
genotypes2 = filter_by_maf(genotypes2, min_maf=0.05)
# Get sample names
smpl1_full = np.array(callset1['samples'])
smpl2_full = np.array(callset2['samples'])
# Iterate random sampling and compute Ho, He, Fis for each iteration
for i in range(n_iter):
    wilcoxon_res = None
    dunn_results = None
    smpl1 = smpl1_full.copy()
    smpl2 = smpl2_full.copy()
    genotypes1_sub, idx1 = subsample_snps(genotypes1, n_target=1600)
    genotypes2_sub, idx2 = subsample_snps(genotypes2, n_target=1600)
    genotypes1_sub, smpl1, kept_indices1 = filter_samples_with_min_snps(genotypes1_sub, smpl1, min_snps=3) # Remove samples with < 3 SNPs
    genotypes2_sub, smpl2, kept_indices2 = filter_samples_with_min_snps(genotypes2_sub, smpl2, min_snps=3) # Remove samples with < 3 SNPs
    # get POP
    pop_df1 = pop_df1_full[pop_df1_full["Sample"].isin(smpl1)].copy()
    pop_df2 = pop_df2_full[pop_df2_full["Sample"].isin(smpl2)].copy()
    samples1 = smpl1
    samples2 = smpl2
    sample_to_pop1 = dict(zip(pop_df1["Sample"], pop_df1["Population"]))
    sample_to_pop2 = dict(zip(pop_df2["Sample"], pop_df2["Population"]))
    populations1 = sorted(pop_df1["Population"].unique())
    populations2 = sorted(pop_df2["Population"].unique())
    subpop1 = group_samples_id_by_pop(samples1, sample_to_pop1, populations1, pop_df1, smpl1)
    subpop2 = group_samples_id_by_pop(samples2, sample_to_pop2, populations2, pop_df2, smpl2)
    all_pop_inclusion.append({
        "iter": i,
        "populations1": list(populations1),
        "populations2": list(populations2),
    })
    # Filter POP
    # delete pop where N < 2
    subpop1, populations1 = delete_small_groups(subpop1, populations1, 2)
    subpop2, populations2 = delete_small_groups(subpop2, populations2, 2)
    # id level summaries
    pop_df1, ind_df1 = unified_ho_he_fis_per_pop(genotypes1_sub, subpop1, populations1, smpl1)
    pop_df2, ind_df2 = unified_ho_he_fis_per_pop(genotypes2_sub, subpop2, populations2, smpl2)
    pop_df = pd.concat([pop_df1, pop_df2], ignore_index=True)
    ind_df = pd.concat([ind_df1, ind_df2], ignore_index=True)
    # Results of the iteration
    if len(subpop1) + len(subpop2) == 2:
        wilcoxon_res = run_wilcoxon_test(ind_df)
    if wilcoxon_res is not None:
        wilcoxon_res['iter'] = i
        all_wilcoxon_results.append(wilcoxon_res)
    if len(subpop1) + len(subpop2) > 2:
        dunn_results = run_dunn_test(ind_df)
    if dunn_results is not None:
        dunn_results['iter'] = i
        all_dunn_results.append(dunn_results)
    stats_df = (
        ind_df.groupby("Population")["Ho"]
        .agg(["count", "mean", "var", "std"])
        .reset_index()
    )
    stats_df['iter'] = i
    all_stats.append(stats_df)
# Aggregate results out of loop here later
wilcoxon_summary = pd.DataFrame(all_wilcoxon_results)
dunn_summary = pd.concat(all_dunn_results, ignore_index=True) if all_dunn_results else None
stats_summary = pd.concat(all_stats, ignore_index=True)
pop_inclusion_summary = pd.DataFrame(all_pop_inclusion)
# Reshape stats_summary to wide format
ho_wide = stats_summary.pivot(index="iter", columns="Population", values=["mean", "std", "var", "count"])
ho_wide.columns = [f"{stat}_{pop}" for stat, pop in ho_wide.columns]
ho_wide = ho_wide.reset_index()
# Auto-detect populations
mean_cols = [c for c in ho_wide.columns if c.startswith("mean_")]
populations_detected = sorted(c.replace("mean_", "", 1) for c in mean_cols)
print(f"Populations detected: {populations_detected}")
# Add Wilcoxon if present
if not wilcoxon_summary.empty:
    master = wilcoxon_summary.merge(ho_wide, on="iter", how="left")
else:
    master = ho_wide.copy()
# All possible deltas
for pop_a, pop_b in itertools.combinations(populations_detected, 2):
    col_name = f"delta_{pop_a}_vs_{pop_b}"
    master[col_name] = master[f"mean_{pop_a}"] - master[f"mean_{pop_b}"]
# Significant or not.
if "p_value" in master.columns:
    master["significant"] = master["p_value"] < 0.05
else:
    master["significant"] = np.nan  # If no Wilcoxon (pop > 2), see Dunn test table instead
#master.to_csv(save_stats, index=False)
#dunn_summary.to_csv(save_dunn, index=False) if dunn_summary is not None else None