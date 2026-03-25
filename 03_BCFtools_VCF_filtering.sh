#!/bin/bash
# Very simple script to filter a VCF to:
# 1) Keep only bi-allelic SNPs (no more than 1 alternative allele)
# 2) Keep only SNPs with a quality superior to 30 (0.1% false-positive probability)
# 3) Keep only SNPs with a depth of at least 5X (5 reads mapped at this position, or more)
vcf=""
out=""
bcftools view --types snps -m 2 -M 2 $vcf -Ou | bcftools filter -i 'QUAL >= 30 && DP >= 5' -Oz -o $out
