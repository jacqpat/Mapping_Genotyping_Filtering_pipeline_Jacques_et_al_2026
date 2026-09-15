library(data.table)
library(vcfR)

is_transition <- function(r, a) {
  (r == "A" & a == "G") |
    (r == "G" & a == "A") |
    (r == "C" & a == "T") |
    (r == "T" & a == "C")
}

gangfile <- "./placeholder.txt"
snpsfile <- "./placeholder.vcf.gz"

groups <- fread(gangfile, header = FALSE)
vcf <- read.vcfR(snpsfile, verbose = TRUE)

colnames(groups) <- c("id","group")
group_map <- setNames(groups$group, groups$id)

ref <- vcf@fix[, "REF"]
alt <- vcf@fix[, "ALT"]

ts_mask <- is_transition(ref, alt)
tv_mask <- !ts_mask

gt <- extract.gt(vcf, element = "GT")
gt <- as.matrix(gt)
valid <- !(gt %in% c("./.", ".", NA))
valid <- matrix(valid, nrow = nrow(gt), ncol = ncol(gt))
sample_ids <- colnames(gt)
sample_groups <- group_map[sample_ids]
# We calculate the titv for each individual then
# We do the mean for each group.
indiv_results <- data.table(
  id = sample_ids,
  group = sample_groups,
  TS = 0,
  TV = 0,
  TiTv = NA_real_
)
for (i in seq_along(sample_ids)) {
  gti <- valid[, i]  # vector of valid genotypes for individual i
  
  ts_count <- sum(gti & ts_mask)
  tv_count <- sum(gti & tv_mask)
  
  indiv_results[i, `:=`(
    TS = ts_count,
    TV = tv_count,
    TiTv = ifelse(tv_count > 0, ts_count / tv_count, NA_real_)
  )]
}
group_results <- indiv_results[, .(
  mean_TiTv = mean(TiTv, na.rm = TRUE),
  sd_TiTv   = sd(TiTv, na.rm = TRUE),
  n_indiv   = .N
), by = group]
print(indiv_results)
print(group_results)
