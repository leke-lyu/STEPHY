#!/usr/bin/env Rscript
#
# Add patristic distance features to edge CSV from edge_feature.py
#
# Input:  edge CSV (graph_id, src, dst, migration_rate) from edge_feature.py
# Output: Same edge CSV updated in-place with patristic_distance column
#
# Usage: Rscript edge_feature.R <edge_csv_file> <tree_file>

library(treeio)
library(ape)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript edge_feature.R <edge_csv_file> <tree_file>\n", file = stderr())
  cat("Example: Rscript edge_feature.R 0_edge.csv 0_beast2.trees\n", file = stderr())
  cat("Note: Updates edge_csv_file in-place with patristic_distance column\n", file = stderr())
  quit(status = 1)
}

edge_csv_file <- args[1]  # Input/Output: edge CSV from edge_feature.py
tree_file <- args[2]       # Input: BEAST2 trees file

# Read edge CSV from edge_feature.py
tryCatch({
  edge_df <- read.csv(edge_csv_file, stringsAsFactors = FALSE)
}, error = function(e) {
  cat("ERROR: Failed to read edge CSV file:", edge_csv_file, "\n", file = stderr())
  cat("Error message:", e$message, "\n", file = stderr())
  quit(status = 1)
})

# Read trees
tryCatch({
  tree <- read.beast(tree_file)
}, error = function(e) {
  cat("ERROR: Failed to read trees file:", tree_file, "\n", file = stderr())
  cat("Error message:", e$message, "\n", file = stderr())
  quit(status = 1)
})

if (is.null(names(tree))) {
  tree <- list(STATE_0 = tree)
}

# Function to get pairwise patristic distances between two location types
get_pairwise_distances <- function(type1, type2, tip_info, patristic_dist) {
  tips_type1 <- tip_info$tip_label[tip_info$type == type1]
  tips_type2 <- tip_info$tip_label[tip_info$type == type2]

  if (length(tips_type1) == 0 || length(tips_type2) == 0) {
    return(numeric(0))
  }

  if (type1 == type2) {
    # Self-loop: pairwise distances within same location
    if (length(tips_type1) > 1) {
      idx <- combn(length(tips_type1), 2)
      patristic_dist[cbind(tips_type1[idx[1,]], tips_type1[idx[2,]])]
    } else {
      # Only one tip: no pairwise distances
      numeric(0)
    }
  } else {
    # Between locations: all pairwise distances
    as.vector(patristic_dist[tips_type1, tips_type2, drop = FALSE])
  }
}

# Build distance cache for all states
# Key format: "state_suffix_min(src,dst)_max(src,dst)" to ensure symmetry
distance_cache <- list()

for (state_name in names(tree)) {
  state_suffix <- sub("STATE_", "", state_name)

  tree_phylo <- as.phylo(tree[[state_name]])
  n_tips <- length(tree_phylo$tip.label)

  # Extract tip information (location types)
  tip_info <- tree[[state_name]]@data %>%
    filter(node %in% 1:n_tips) %>%
    arrange(as.numeric(node)) %>%
    transmute(
      tip_number = as.numeric(node),
      tip_label = tree_phylo$tip.label[as.numeric(node)],
      type = gsub("I\\{|\\}", "", type)  # Remove I{ and }
    )

  # Calculate patristic distance matrix
  patristic_dist <- cophenetic.phylo(tree_phylo)

  # Get all unique location types
  all_types <- sort(unique(tip_info$type))

  # Calculate distances for all location pairs (including self-loops)
  for (src in all_types) {
    for (dst in all_types) {
      # Use sorted pair to ensure A->B and B->A use same cache key
      sorted_pair <- sort(c(src, dst))
      cache_key <- paste(state_suffix, sorted_pair[1], sorted_pair[2], sep = "_")

      # Skip if already calculated (ensures symmetry)
      if (cache_key %in% names(distance_cache)) {
        next
      }

      # Calculate distances
      distances <- get_pairwise_distances(src, dst, tip_info, patristic_dist)

      if (length(distances) > 0) {
        sorted_distances <- sort(distances)
        distance_str <- paste0("[", paste(sorted_distances, collapse = "\t"), "]")
      } else {
        # No distances: use NA instead of "[]" for consistency
        distance_str <- NA
      }

      distance_cache[[cache_key]] <- distance_str
    }
  }
}

# Add patristic_distance column to edge_df
# Pre-extract columns for efficiency
graph_ids <- edge_df$graph_id
srcs <- as.character(edge_df$src)
dsts <- as.character(edge_df$dst)

edge_df$patristic_distance <- sapply(seq_len(nrow(edge_df)), function(i) {
  # Extract state suffix from graph_id (e.g., "0_5" -> "5")
  state_suffix <- sub("^.*_", "", graph_ids[i])

  # Use sorted pair to ensure symmetry (A->B and B->A get same distances)
  sorted_pair <- sort(c(srcs[i], dsts[i]))
  cache_key <- paste(state_suffix, sorted_pair[1], sorted_pair[2], sep = "_")

  if (cache_key %in% names(distance_cache)) {
    distance_cache[[cache_key]]
  } else {
    NA  # NA if no distances found (consistent with migration_rate)
  }
})

# Write updated CSV back to same file (in-place update)
# Use na = "" to write NaN as empty cells (consistent with Python pandas)
write.csv(edge_df, file = edge_csv_file, row.names = FALSE, quote = FALSE, na = "")

cat("Successfully added patristic_distance column to", nrow(edge_df), "edges\n")
cat("Updated file:", edge_csv_file, "\n")
