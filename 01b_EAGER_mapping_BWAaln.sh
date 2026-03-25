#!/bin/bash
#SBATCH -J eager_Ssal_rescaled
#SBATCH -p unlimitq

module purge
module load bioinfo/NextflowWorkflows/nfcore-Nextflow-v21.10.6
module load containers/singularity/3.9.9


path_to_file=""
path_to_ref=""
path_to_config=""
name_of_file="" # .FASTQ.gz. Example if you have both forward and reverse reads: *{1,2}*.fastq.gz
name_of_ref="" # .fa (or FASTA, etc.)
name_of_idx="" # .fa.fai
name_of_seq="" # .dict
name_of_config="" # .config

nextflow run nf-core/eager \
    --input '${path_to_file}/${name_of_file}' \
    --fasta '${path_to_ref}/${name_of_ref}' \
    --fasta_index '${path_to_file}/${name_of_idx}' \
    --seq_dict '${path_to_ref}/${name_of_seq}' \
    -c '${path_to_config}/${name_of_config}' \
    --outdir './' \
    -profile genotoul \
    --skip_fastqc \
    --skip_adapterremoval \
    --clip_readlength 25 \
    --mapper 'bwaaln' \
    --mergedonly \
    --dedupper 'dedup' \
    --dedup_all_merged \
    --run_mapdamage_rescaling
