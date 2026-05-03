library(ggplot2)

base_dir <- "C:/Users/willi/OneDrive/Documents/University/Year 3/Semester 2/Capstone/Designated Species"
out_dir <- file.path(base_dir, "Final Figures")

df <- read.csv(file.path(base_dir, "master_species_data.csv"))

bases <- c("A", "C", "G", "T")
mut_types_12 <- c("A>C", "A>G", "A>T", "C>A", "C>G", "C>T",
                  "G>A", "G>C", "G>T", "T>A", "T>C", "T>G")

# 6 collapsed mutation types (reverse-complement pairs)
collapse_pairs <- list(
  "C>A" = c("C>A", "G>T"),
  "C>G" = c("C>G", "G>C"),
  "C>T" = c("C>T", "G>A"),
  "T>A" = c("T>A", "A>T"),
  "T>C" = c("T>C", "A>G"),
  "T>G" = c("T>G", "A>C")
)
collapsed_types <- names(collapse_pairs)
title_map_6 <- list(
  "C>A" = "C>A / G>T",
  "C>G" = "C>G / G>C",
  "C>T" = "C>T / G>A",
  "T>A" = "T>A / A>T",
  "T>C" = "T>C / A>G",
  "T>G" = "T>G / A>C"
)
opp_pair <- list(
  "C>A" = c("C", "G"),
  "C>G" = c("C", "G"),
  "C>T" = c("C", "G"),
  "T>A" = c("T", "A"),
  "T>C" = c("T", "A"),
  "T>G" = c("T", "A")
)

# Format species names: "Acinetobacter_baumannii" -> "A. baumannii"
species_labels <- sapply(strsplit(df$Species, "_"), function(x) {
  if (length(x) >= 2) {
    paste0(substr(x[1], 1, 1), ". ", paste(x[-1], collapse = " "))
  } else {
    x[1]
  }
})

# Distinct 18-colour palette
species_palette <- c(
  "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
  "#911EB4", "#42D4F4", "#F032E6", "#9A6324", "#FABED4",
  "#469990", "#DCBEFF", "#800000", "#AAFFC3", "#808000",
  "#000075", "#A9A9A9", "#000000"
)

build_long_12 <- function(df) {
  rows <- list()
  for (i in seq_len(nrow(df))) {
    sp <- species_labels[i]
    total <- df$total_singletons[i]
    nr_lookup <- c("A" = df$nr_genome_A[i], "C" = df$nr_genome_C[i],
                   "G" = df$nr_genome_G[i], "T" = df$nr_genome_T[i])
    nr_total <- sum(nr_lookup)
    if (total == 0 || nr_total == 0) next
    
    for (mt in mut_types_12) {
      ref <- substr(mt, 1, 1)
      alt <- substr(mt, 3, 3)
      observed_count <- df[[paste0("singleton_count_", ref, "_to_", alt)]][i]
      observed_pct <- df[[paste0("singleton_pct_", ref, "_to_", alt)]][i]
      
      expected_count <- total * (nr_lookup[ref] / nr_total) * (1 / 3)
      expected_pct <- expected_count / total * 100
      
      if (expected_count > 0) {
        if (observed_count > expected_count) {
          p <- 1 - ppois(observed_count - 1, expected_count)
        } else {
          p <- ppois(observed_count, expected_count)
        }
        p <- min(p * 2, 1)
      } else {
        p <- 1
      }
      p_bonf <- min(p * 12, 1)
      stars <- if (p_bonf < 0.001) "***" else if (p_bonf < 0.01) "**" else if (p_bonf < 0.05) "*" else ""
      
      rows[[length(rows) + 1]] <- data.frame(
        species = sp, mutation_type = mt,
        observed_pct = observed_pct, expected_pct = expected_pct,
        stars = stars, stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

build_long_6 <- function(df) {
  rows <- list()
  for (i in seq_len(nrow(df))) {
    sp <- species_labels[i]
    nr_lookup <- c("A" = df$nr_genome_A[i], "C" = df$nr_genome_C[i],
                   "G" = df$nr_genome_G[i], "T" = df$nr_genome_T[i])
    nr_total <- sum(nr_lookup)
    if (nr_total == 0) next
    
    obs_lookup <- list()
    for (mt in mut_types_12) {
      ref <- substr(mt, 1, 1)
      alt <- substr(mt, 3, 3)
      obs_lookup[[mt]] <- df[[paste0("singleton_count_", ref, "_to_", alt)]][i]
    }
    
    collapsed_obs <- sapply(collapsed_types, function(ct) {
      sum(unlist(obs_lookup[collapse_pairs[[ct]]]))
    })
    total_obs <- sum(collapsed_obs)
    if (total_obs == 0) next
    
    for (ct in collapsed_types) {
      bs <- opp_pair[[ct]]
      opp_sum <- sum(nr_lookup[bs])
      expected_count <- total_obs * (opp_sum / nr_total) * (1 / 3)
      expected_pct <- expected_count / total_obs * 100
      observed_count <- collapsed_obs[ct]
      observed_pct <- observed_count / total_obs * 100
      
      if (expected_count > 0) {
        if (observed_count > expected_count) {
          p <- 1 - ppois(observed_count - 1, expected_count)
        } else {
          p <- ppois(observed_count, expected_count)
        }
        p <- min(p * 2, 1)
      } else {
        p <- 1
      }
      p_bonf <- min(p * 6, 1)
      stars <- if (p_bonf < 0.001) "***" else if (p_bonf < 0.01) "**" else if (p_bonf < 0.05) "*" else ""
      
      rows[[length(rows) + 1]] <- data.frame(
        species = sp, mutation_type = title_map_6[[ct]],
        observed_pct = observed_pct, expected_pct = expected_pct,
        stars = stars, stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

plot_grid <- function(plot_df, mt_levels, ncol_facets, out_path, plot_w, plot_h) {
  plot_df$species <- factor(plot_df$species, levels = unique(species_labels))
  plot_df$mutation_type <- factor(plot_df$mutation_type, levels = mt_levels)
  
  ymax <- max(plot_df$observed_pct, na.rm = TRUE) * 1.18
  
  p <- ggplot(plot_df, aes(x = species, y = observed_pct, fill = species)) +
    geom_col(width = 0.8, colour = "grey25", linewidth = 0.18) +
    geom_segment(
      aes(x = as.numeric(species) - 0.42,
          xend = as.numeric(species) + 0.42,
          y = expected_pct,
          yend = expected_pct),
      colour = "black", linewidth = 0.55, inherit.aes = FALSE,
      data = plot_df
    ) +
    geom_text(aes(label = stars, y = observed_pct + ymax * 0.015),
              size = 2.6, vjust = 0) +
    facet_wrap(~ mutation_type, ncol = ncol_facets, scales = "fixed", axes = "all") +
    scale_y_continuous(
      name = "Singletons (% of species total)",
      limits = c(0, ymax),
      expand = c(0, 0)
    ) +
    scale_x_discrete(name = NULL) +
    scale_fill_manual(values = species_palette, name = "Species") +
    theme_classic(base_size = 11) +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.line.x = element_line(colour = "grey30", linewidth = 0.4),
      axis.text.y = element_text(size = 9, colour = "grey25", angle = 90, hjust = 0.5),
      axis.title.y = element_text(size = 12, margin = margin(r = 8)),
      axis.line.y = element_line(colour = "grey30", linewidth = 0.4),
      axis.ticks.y = element_line(colour = "grey30", linewidth = 0.3),
      panel.spacing.x = unit(1.2, "lines"),
      panel.spacing.y = unit(1, "lines"),
      strip.text = element_text(size = 11, face = "bold", margin = margin(b = 5, t = 2)),
      strip.background = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 9, face = "italic"),
      legend.key.size = unit(0.45, "cm"),
      legend.spacing.x = unit(0.3, "cm"),
      legend.spacing.y = unit(0.1, "cm"),
      plot.margin = margin(t = 8, r = 8, b = 8, l = 8)
    ) +
    guides(fill = guide_legend(nrow = 3, byrow = TRUE))
  
  ggsave(out_path, p, width = plot_w, height = plot_h, dpi = 400, bg = "white")
  cat("Saved: ", out_path, "\n", sep = "")
}

plot_df_12 <- build_long_12(df)
plot_grid(
  plot_df_12, mut_types_12, 4,
  file.path(out_dir, "singleton_spectrum_grid_12.png"),
  14, 10
)

plot_df_6 <- build_long_6(df)
mt_levels_6 <- unname(unlist(title_map_6[collapsed_types]))
plot_grid(
  plot_df_6, mt_levels_6, 3,
  file.path(out_dir, "singleton_spectrum_grid_6.png"),
  13, 8
)