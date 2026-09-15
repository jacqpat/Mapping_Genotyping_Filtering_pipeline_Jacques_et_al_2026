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

def ho_and_missing(genotypes, subpops, populations, samples):
    ind_results = []
    for pop, indices in zip(populations, subpops):
        geno_sub = genotypes[:, indices]
        # Precompute masks
        het = geno_sub.is_het()          # heterozygous calls
        called = geno_sub.is_called()    # non-missing calls
        missing = geno_sub.is_missing()  # missing calls
        for i, idx in enumerate(indices):
            # Called genotypes
            n_called_i = np.sum(called[:, i])
            # Observed heterozygosity
            if n_called_i > 0:
                ho_i = np.sum(het[:, i] & called[:, i]) / n_called_i
            else:
                ho_i = np.nan
            # Missingness
            missing_i = np.sum(missing[:, i]) / genotypes.shape[0]
            # Results
            ind_results.append({
                "Sample": samples[idx],
                "Population": pop,
                "Ho": ho_i,
                "Missingness": missing_i
            })
    return pd.DataFrame(ind_results)

def get_miss_ho(callset, pop_df, min_maf=0.05, min_snps=3, prefix=""):
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    genotypes = filter_by_maf(genotypes, min_maf=min_maf)
    smpl = callset['samples']
    print("Filtering samples with not enough SNPs...")
    genotypes, smpl, kept_indices = filter_samples_with_min_snps(genotypes, smpl, min_snps=min_snps)
    pop_df = pop_df[pop_df["Sample"].isin(smpl)]
    print("Add species to population names...")
    pop_df["Population"] = pop_df["Population"].apply(lambda x: f"{prefix}{x}")
    samples = smpl
    sample_to_pop = dict(zip(pop_df["Sample"], pop_df["Population"]))
    populations = sorted(pop_df["Population"].unique())
    print("Grouping samples by population...")
    subpop = group_samples_id_by_pop(samples, sample_to_pop, populations, pop_df, smpl)
    print("Computing Ho and missingness...")
    df = ho_and_missing(genotypes, subpop, populations, samples)
    return df

callset1=allel.read_vcf(r"") # .vcf.gz
pop_df1= pd.read_csv(r"", sep="\t", header=None, names=["Sample", "Population"]) # .txt with two columns: sample names and their population
print("Reading VCFs...")
df1 = get_miss_ho(callset1, pop_df1, min_maf=0.05, min_snps=3, prefix="anguilla-")
# if there's more than one dataframe. Then create callset2 + pop_df2, etc... for each of your vcf + population file pair and add them to the df_all list.
df_all = pd.concat([df1], ignore_index=True)
print("Computing Spearman correlation between Ho and Missingness...")
for pop in df_all["Population"].unique():
    sub = df_all[df_all["Population"] == pop]
    rho, p = stats.spearmanr(sub["Missingness"], sub["Ho"])
    print(f"{pop}: Spearman rho={rho:.3f}, p={p:.3g}")
print("Plotting Ho vs Missingness...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_all,
    x="Missingness",
    y="Ho",
    hue="Population",
    s=80
)
for pop in df_all["Population"].unique():
    sub = df_all[df_all["Population"] == pop]
    sns.regplot(
        data=sub,
        x="Missingness",
        y="Ho",
        scatter=False,
        label=pop,
        line_kws={"linewidth": 2}
    )
plt.xlabel("Missingness")
plt.ylabel("Observed Heterozygosity (Ho)")
plt.grid(True)
plt.tight_layout()
plt.show()
