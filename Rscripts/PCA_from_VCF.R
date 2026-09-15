library(adegenet)
library(vcfR)
library(tidyr)
library(dplyr)
library(purrr)
library(dartR)
library(ggplot2)
library(cowplot)
library(RColorBrewer)
library(ade4)
library(ggrepel)
set.seed(42)
# load metadata
x_info<-read.table("./placeholder.txt", header=F)
colnames(x_info) <- c("id","pop")
# load VCF - convert to Genlight
vcf <- read.vcfR("./placeholder.vcf.gz")
genlight_obj <- vcfR2genlight(vcf)
pop(genlight_obj) <- x_info$pop[match(indNames(genlight_obj), x_info$id)]
indNames(genlight_obj) <- x_info$id[match(indNames(genlight_obj), x_info$id)]
# Genlight - remove orphan samples
pop_sizes <- table(pop(genlight_obj))
keep_pops <- names(pop_sizes[pop_sizes >= 2])
genlight_obj <- genlight_obj[pop(genlight_obj) %in% keep_pops, ]
# Genlight - imputation
X <- tab(genlight_obj, NA.method="mean")
# Remove loci with any NA
X_clean <- X[, colSums(is.na(X)) == 0]
# PCA
pca_results <- prcomp(X_clean, center=TRUE, scale.=FALSE)
# PCA - results as dataframe
pca_df <- data.frame(pca_results$x[,1:3], pop=pop(genlight_obj), id=indNames(genlight_obj))
colnames(pca_df)[1:3] <- c("PC1","PC2","PC3")
# PCA - plot
library(viridis)
ggplot(pca_df, aes(x=PC1, y=PC2, color=pop, fill = pop, shape=pop)) +
  geom_point(size=3, alpha=0.8) +
  xlim(min(pca_df$PC1) - 5, max(pca_df$PC1) + 5) +
  ylim(min(pca_df$PC2) - 5, max(pca_df$PC2) + 5) +
  theme_minimal(base_size=14) +
  # (it's not exactly clean but at least by repeating 21 to 25 until you have
  # more values than groups (pop), then you'll have a shape for each group and no error)
  scale_shape_manual(values = c(21,22,23,24,25,21,22,23,24,25,21,22,23,24,25,21,22,23,24,25)) +
  scale_color_viridis_d(option = "turbo") +
  scale_fill_viridis_d(option = "turbo") +
  labs(x=paste0("PC1 (", round(100*summary(pca_results)$importance[2,1],1), "%)"),
       y=paste0("PC2 (", round(100*summary(pca_results)$importance[2,2],1), "%)"))

# BONUS - save Genlight
saveRDS(genlight_obj, "./placeholder.rds")
