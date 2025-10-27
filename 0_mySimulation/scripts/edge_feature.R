#!/usr/bin/env Rscript

library(treeio)
library(ape)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
tree_file <- args[1]
output_file <- args[2]

tree <- read.beast(tree_file)

file_prefix <- sub("_beast2\\.trees$", "", basename(tree_file))

if (is.null(names(tree))) {
  tree <- list(STATE_0 = tree)
}

get_pairwise_distances <- function(type1, type2, tip_info, patristic_dist) {
  tips_type1 <- tip_info$tip_label[tip_info$type == type1]
  tips_type2 <- tip_info$tip_label[tip_info$type == type2]

  if (length(tips_type1) == 0 || length(tips_type2) == 0) {
    return(numeric(0))
  }

  if (type1 == type2 && length(tips_type1) > 1) {
    idx <- combn(length(tips_type1), 2)
    patristic_dist[cbind(tips_type1[idx[1,]], tips_type1[idx[2,]])]
  } else if (type1 != type2) {
    as.vector(patristic_dist[tips_type1, tips_type2, drop = FALSE])
  } else {
    numeric(0)
  }
}

all_csv_data <- list()

for (state_name in names(tree)) {
  state_suffix <- sub("STATE_", "", state_name)
  graph_id <- paste0(file_prefix, "_", state_suffix)

  tree_phylo <- as.phylo(tree[[state_name]])
  n_tips <- length(tree_phylo$tip.label)

  tip_info <- tree[[state_name]]@data %>%
    filter(node %in% 1:n_tips) %>%
    arrange(as.numeric(node)) %>%
    transmute(
      tip_number = as.numeric(node),
      tip_label = tree_phylo$tip.label[as.numeric(node)],
      type = gsub("I\\{", "", type)
    )

  all_types <- sort(unique(tip_info$type))

  patristic_dist <- cophenetic.phylo(tree_phylo)

  type_combinations <- c(
    lapply(all_types, function(t) c(t, t)),
    if (length(all_types) > 1) combn(all_types, 2, simplify = FALSE) else list()
  )

  csv_data <- do.call(rbind, lapply(type_combinations, function(combo) {
    distances <- get_pairwise_distances(combo[1], combo[2], tip_info, patristic_dist)

    if (length(distances) > 0) {
      sorted_distances <- sort(distances)
      data.frame(
        graph_id = graph_id,
        node1 = combo[1],
        node2 = combo[2],
        edge_features = paste0("[", paste(sorted_distances, collapse = "\t"), "]"),
        stringsAsFactors = FALSE
      )
    } else {
      NULL
    }
  }))

  all_csv_data[[length(all_csv_data) + 1]] <- csv_data
}

final_csv_data <- do.call(rbind, all_csv_data)

write.table(final_csv_data,
            file = output_file,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE,
            quote = FALSE)

cat("Successfully processed", length(all_csv_data), "state(s)\n")
cat("Output written to:", output_file, "\n")
