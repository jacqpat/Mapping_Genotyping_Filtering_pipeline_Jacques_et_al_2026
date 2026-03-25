#!/bin/bash

vcf1=""  # first VCF file
vcf2=""  # second VCF file
vcfo=""  # intermediary file: 1rst VCF' samples W/ only shared SNPs.
vcfp=""  # intermediary file: 2nd VCF' samples W/ only shared SNPs.
vcff=""  # final output name
bcftools isec -n=2 -w1 $vcf1 $vcf2 -Oz -o $vcfo
echo "indexing one..."
bcftools index $vcfo
bcftools isec -n=2 -w1 $vcf2 $vcf1 -Oz -o $vcfp
echo "indexing two..."
bcftools index $vcfp
bcftools merge $vcfo $vcfp -Oz -o $vcff
echo "merge done"
