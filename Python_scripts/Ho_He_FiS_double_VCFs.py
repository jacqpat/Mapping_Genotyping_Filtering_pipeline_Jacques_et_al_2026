import sys
import allel
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

def union_genotypes(callset1, callset2,g1,g2):
    # Load variant positions (chrom + pos)
    # Adjust keys if your VCF uses different names
    pos1 = callset1['variants/POS']
    pos2 = callset2['variants/POS']
    chrom1 = callset1['variants/CHROM']
    chrom2 = callset2['variants/CHROM']
    # Build a structured array of (chrom, pos)
    v1 = np.core.records.fromarrays([chrom1, pos1], names='chrom,pos')
    v2 = np.core.records.fromarrays([chrom2, pos2], names='chrom,pos')
    # Union of variants
    union_variants = np.union1d(v1, v2)
    # Prepare empty genotype matrices
    # Shape: (n_variants_union, n_samples, ploidy)
    ploidy = g1.shape[2]
    # assuming both have same ploidy
    g1_aligned = np.full((len(union_variants), g1.shape[1], ploidy), -1, dtype='i1')
    g2_aligned = np.full((len(union_variants), g2.shape[1], ploidy), -1, dtype='i1')
    # Fill in g1
    # Find indices of v1 inside union_variants
    idx1 = np.where(np.in1d(union_variants, v1))[0]
    g1_aligned[idx1, :, :] = g1
    # Fill in g2
    idx2 = np.where(np.in1d(union_variants, v2))[0]
    g2_aligned[idx2, :, :] = g2
    # Concatenate samples
    return allel.GenotypeArray( np.concatenate([g1_aligned, g2_aligned], axis=1) )

callset1=allel.read_vcf(r"")    # VCF or VCF.GZ
callset2=allel.read_vcf(r"")    # VCF or VCF.GZ
pop_df1= pd.read_csv(r"", sep="\t", header=None, names=["Sample", "Population"])    # Headless TSV
pop_df2= pd.read_csv(r"", sep="\t", header=None, names=["Sample", "Population"])    # Headless TSV
order = ["Placeholder_X","Placeholder_Y","Placeholder_Z"]    # Order of the populations shown in the final Boxplot graph
dunn_save=r""    # CSV
hohe_save=r""    # CSV
boxx_save=r""    # SVG
stat_save=r""    # CSV
genotypes1 = allel.GenotypeArray(callset1['calldata/GT'])
genotypes2 = allel.GenotypeArray(callset2['calldata/GT'])
# filters VCF
genotypes1 = filter_by_maf(genotypes1, min_maf=0.05)    # You can change min_maf according to your needs
genotypes2 = filter_by_maf(genotypes2, min_maf=0.05)    # You can change min_maf according to your needs
smpl1 = callset1['samples']
smpl2 = callset2['samples']
genotypes1, smpl1, kept_indices1 = filter_samples_with_min_snps(genotypes1, smpl1, min_snps=3)    # Remove samples with < 3 SNPs
genotypes2, smpl2, kept_indices2 = filter_samples_with_min_snps(genotypes2, smpl2, min_snps=3)    # Remove samples with < 3 SNPs
print("Read POP")
pop_df1 = pop_df1[pop_df1["Sample"].isin(smpl1)]        # Match samples directly
pop_df2 = pop_df2[pop_df2["Sample"].isin(smpl2)]        # Match samples directly
samples1 = smpl1
samples2 = smpl2
sample_to_pop1 = dict(zip(pop_df1["Sample"], pop_df1["Population"]))
sample_to_pop2 = dict(zip(pop_df2["Sample"], pop_df2["Population"]))
populations1 = sorted(pop_df1["Population"].unique())
populations2 = sorted(pop_df2["Population"].unique())
subpop1 = group_samples_id_by_pop(samples1, sample_to_pop1, populations1, pop_df1, smpl1)
subpop2 = group_samples_id_by_pop(samples2, sample_to_pop2, populations2, pop_df2, smpl2)
print("Filter POP")
# delete pop where N < 2
subpop1, populations1 = delete_small_groups(subpop1, populations1, 2)
subpop2, populations2 = delete_small_groups(subpop2, populations2, 2)
# id level summaries
pop_df1, ind_df1 = unified_ho_he_fis_per_pop(genotypes1, subpop1, populations1, smpl1)
pop_df2, ind_df2 = unified_ho_he_fis_per_pop(genotypes2, subpop2, populations2, smpl2)
pop_df = pd.concat([pop_df1, pop_df2], ignore_index=True)
ind_df = pd.concat([ind_df1, ind_df2], ignore_index=True)
pop_df.to_csv(hohe_save, index=False)
dunn_results = run_dunn_test(ind_df)
if dunn_results is not None:
    print("\nDunn test pairwise p-values:")
    print(dunn_results)
    dunn_results.to_csv(dunn_save)
stats_df = (
    ind_df
    .groupby("Population")["Ho"]
    .agg(["count","mean","var","std"])
    .reset_index()
)
print(stats_df)
# Boxplot
stats_df.to_csv(stat_save, index=False)
ind_df["Population"] = pd.Categorical(ind_df["Population"], ordered=True, categories = order)
mpl.rcParams['svg.fonttype'] = 'none'
plt.figure(figsize=(8, 8))
sns.boxplot(data=ind_df, x="Population", y="Ho", hue="Population",palette="Set2", showfliers = False, legend=False, dodge=False)
sns.stripplot(data=ind_df, x="Population", y="Ho", color="black", alpha=0.6, jitter=True)
plt.ylabel("Observed Heterozygosity (Ho)")
plt.xlabel("Population")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(boxx_save, format="svg", transparent=True, bbox_inches="tight")
plt.show()
