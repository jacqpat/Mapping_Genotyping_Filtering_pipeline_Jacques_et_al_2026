import sys
import allel
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib as mpl
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

def unified_ho_he_fis_per_pop(genotypes, subpops, populations):
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
            ind_results.append({ "Sample": idx, "Population": pop, "Ho": ho_i })
        ho_pop = np.nanmean(ho_individuals)
        # --- Wright's FIS ---
        if he_pop > 0 and np.isfinite(he_pop):
            fis_pop = (he_pop - ho_pop) / he_pop
        else:
            fis_pop = np.nan
        pop_results.append({ "Population": pop, "Ho": ho_pop, "He": he_pop, "FIS": fis_pop, "N": len(indices)})
    return pd.DataFrame(pop_results), pd.DataFrame(ind_results)

def run_dunn_test(ho_df):
    groups = [group["Ho"].dropna().values for _, group in ho_df.groupby("Population")]
    # Run Bartlett test for variance
    bart_stat, bart_p = stats.bartlett(*groups)
    print(f"Bartlett test: χ²={bart_stat:.3f}, p={bart_p:.4f}")
    if bart_p < 0.05:
        print("Variances differ significantly between populations.")
    else:
        print("Variances don't differ significantly between populations.")
    # Run Kruskal-Wallis
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H={stat:.3f}, p={p:.4f}")
    if p < 0.05:
        print("Significant differences detected. Running Dunn test...")
        dunn = sp.posthoc_dunn(ho_df, val_col="Ho", group_col="Population", p_adjust="bonferroni")
        return dunn
    else:
        print("No significant differences between populations.")
        return None

### MAIN ###
inds_save=r""    # CSV
stat_save=r""    # CSV
dunn_save=r""    # CSV
hohe_save=r""    # CSV
boxx_save=r""    # SVG
callset = allel.read_vcf(r"")    # VCF or VCF.GZ
pop_df = pd.read_csv(r"", sep="\t", header=None, names=["Sample", "Population"])    # Headless TSV
order = ["Placeholder_X","Placeholder_Y"]    # The order in which you want your populations to appear in the final Boxplot.
print("Read VCF")
genotypes = allel.GenotypeArray(callset['calldata/GT'])
print("Filter VCF")
genotypes = filter_by_maf(genotypes, min_maf=0.05)    # Change min_maf if you want to be more or less strict.
smpl = callset['samples']
genotypes, smpl, kept_indices = filter_samples_with_min_snps(genotypes, smpl, min_snps=3)    # Remove samples with < 3 SNPs
print("Read POP")
pop_df = pop_df[pop_df["Sample"].isin(smpl)]        # Match samples directly
samples = smpl
sample_to_pop = dict(zip(pop_df["Sample"], pop_df["Population"]))
populations = sorted(pop_df["Population"].unique())
# Group sample indices by population
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
print("Filter POP")
# Addendum : eliminate populations that are smaller than 2
valid = [i for i, inds in enumerate(subpops) if len(inds) >= 2]
subpops = [subpops[i] for i in valid]
populations = [populations[i] for i in valid]
# Phase 1.a: individual level summaries
pop_df, ind_df = unified_ho_he_fis_per_pop(genotypes, subpops, populations)
# Phase 1.b: population level summaries
print(pop_df)
pop_df.to_csv(hohe_save, index=False)
ind_df.to_csv(inds_save, index=False)
# phase 2.a: Dunn test
dunn_results = run_dunn_test(ind_df)
if dunn_results is not None:
    print("\nDunn test pairwise p-values:")
    print(dunn_results)
    dunn_results.to_csv(dunn_save)
# phase 2.b: Statistics
stats_df = (
    ind_df
    .groupby("Population")["Ho"]
    .agg(["count","mean","var","std"])
    .reset_index()
)
print(stats_df)
stats_df.to_csv(stat_save, index=False)
# phase 3: Visualisation
ind_df["Population"] = pd.Categorical(ind_df["Population"], ordered=True, categories = order)
mpl.rcParams['svg.fonttype'] = 'none'
plt.figure(figsize=(8, 8))
sns.boxplot(data = ind_df, x = "Population", y = "Ho", hue = "Population",palette = "Set2", showfliers = False, legend=False, dodge=False)
sns.stripplot(data = ind_df, x = "Population", y = "Ho", color = "black", alpha=0.6, jitter=True)
plt.ylim(0.15, 0.45)
plt.ylabel("Observed Heterozygosity (Ho)")
plt.xlabel("")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(boxx_save, format="svg", transparent=True, bbox_inches="tight")
plt.show()
