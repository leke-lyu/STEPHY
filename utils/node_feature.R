#!/usr/bin/env Rscript
# Add phylogeny-based features to node CSV
# Usage: Rscript node_feature.R <node_csv_file> <tree_file>

library(treeio)
library(ape)
library(phangorn)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript node_feature.R <node_csv_file> <tree_file>\n", file = stderr())
  quit(status = 1)
}

node_csv_file <- args[1]
tree_file <- args[2]

node_df <- read.csv(node_csv_file, stringsAsFactors = FALSE)
tree <- read.beast(tree_file)
if (is.null(names(tree))) tree <- list(STATE_0 = tree)

count_monophyletic_groups <- function(tree_phylo, tip_info, location_type) {
  tips_of_type <- tip_info$tip_number[tip_info$type == location_type]
  if (length(tips_of_type) <= 1) return(length(tips_of_type))

  unassigned_tips <- tips_of_type
  group_count <- 0

  while (length(unassigned_tips) > 0) {
    current_node <- unassigned_tips[1]

    while (TRUE) {
      parent_node <- tree_phylo$edge[tree_phylo$edge[, 2] == current_node, 1]
      if (length(parent_node) == 0) break

      parent_descendants <- Descendants(tree_phylo, parent_node, type = "tips")[[1]]
      if (all(parent_descendants %in% tips_of_type)) {
        current_node <- parent_node
      } else {
        break
      }
    }

    group_tips <- if (current_node <= length(tree_phylo$tip.label)) {
      current_node
    } else {
      Descendants(tree_phylo, current_node, type = "tips")[[1]]
    }

    unassigned_tips <- setdiff(unassigned_tips, group_tips)
    group_count <- group_count + 1
  }

  return(group_count)
}

# Build caches for all phylogeny-based features
monophyletic_cache <- list()
earliest_time_cache <- list()
median_time_cache <- list()
latest_time_cache <- list()

for (state_name in names(tree)) {
  state_suffix <- sub("STATE_", "", state_name)
  tree_phylo <- as.phylo(tree[[state_name]])
  n_tips <- length(tree_phylo$tip.label)

  tip_info <- tree[[state_name]]@data %>%
    filter(node %in% 1:n_tips) %>%
    arrange(as.numeric(node)) %>%
    transmute(
      tip_number = as.numeric(node),
      type = gsub("I\\{|\\}", "", type),
      height = as.numeric(time)
    )

  for (location in unique(tip_info$type)) {
    cache_key <- paste(state_suffix, location, sep = "_")
    tips_of_type <- tip_info[tip_info$type == location, ]

    monophyletic_cache[[cache_key]] <- count_monophyletic_groups(tree_phylo, tip_info, location)
    earliest_time_cache[[cache_key]] <- min(tips_of_type$height, na.rm = TRUE)
    median_time_cache[[cache_key]] <- median(tips_of_type$height, na.rm = TRUE)
    latest_time_cache[[cache_key]] <- max(tips_of_type$height, na.rm = TRUE)
  }
}

# Remove old columns if re-running
old_cols <- c("monophyletic_groups", "Monophyletic_Groups",
              "Earliest_Sample_Time", "Median_Sample_Time", "Latest_Sample_Time")
node_df <- node_df[, !(names(node_df) %in% old_cols), drop = FALSE]

# Add phylogeny-based columns
lookup <- function(cache, graph_id, node) {
  state_suffix <- sub("^.*_", "", graph_id)
  cache_key <- paste(state_suffix, node, sep = "_")
  if (cache_key %in% names(cache)) cache[[cache_key]] else NA
}

node_df$Monophyletic_Groups <- mapply(lookup, list(monophyletic_cache), node_df$graph_id, node_df$node)
node_df$Earliest_Sample_Time <- mapply(lookup, list(earliest_time_cache), node_df$graph_id, node_df$node)
node_df$Median_Sample_Time <- mapply(lookup, list(median_time_cache), node_df$graph_id, node_df$node)
node_df$Latest_Sample_Time <- mapply(lookup, list(latest_time_cache), node_df$graph_id, node_df$node)

write.csv(node_df, file = node_csv_file, row.names = FALSE, quote = FALSE, na = "")
cat("Added 4 phylogeny columns to", nrow(node_df), "nodes ->", node_csv_file, "\n")
