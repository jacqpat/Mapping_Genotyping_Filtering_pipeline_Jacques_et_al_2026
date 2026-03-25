#!/bin/bash

module load bioinfo/Bcftools/1.17

ulimit -n 2048 # changed ulimit because we had a lot of bam files.

bam="list_all_bam.txt" # a simple .txt file with all your .bam files, one file per row.
ref="" # FASTA (.fa, .fasta, etc.)
out="" # .vcf
bcftools mpileup -f $ref -b $bam -q 20 -Q 20 -Ou | bcftools call -mv -Oz -o $out
