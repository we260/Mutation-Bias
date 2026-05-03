library(ggplot2)

base_dir <- "C:/Users/willi/OneDrive/Documents/University/Year 3/Semester 2/Capstone/Designated Species"
out_dir <- file.path(base_dir, "Final Figures")

species <- c(
  'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
  'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
  'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
  'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
  'V_cholerae', 'S_typhimurium_2'
)

labels <- c(
  "A. baumannii", "B. pertussis", "C. jejuni", "C. difficile", "E. coli",
  "H. influenzae", "L. monocytogenes", "M. tuberculosis",
  "N. gonorrhoeae", "N. meningitidis", "P. aeruginosa", "S. aureus",
  "S. epidermidis", "S. agalactiae", "S. pneumoniae", "S. pyogenes",
  "V. cholerae", "S. typhimurium"
)

# Load all per-species GC skew data into one long-format table
all_data <- list()

for (i in seq_along(species)) {
  sp <- species[i]
  path <- file.path(base_dir, sp, "results", "gc_skew_windows.csv")
  if (!file.exists(path)) {
    cat("Missing:", sp, "\n")
    next
  }
  d <- read.csv(path)
  d$species_folder <- sp
  d$species <- labels[i]
  all_data[[sp]] <- d
}

# Compute recombinant percentage per species and build the facet label
recomb_pct <- sapply(all_data, function(d) {
  round(mean(d$is_recombinant) * 100, 1)
})
strip_labels <- paste0(labels, " (", recomb_pct[species], "%)")
names(strip_labels) <- labels

# Apply the strip label as the species level
combined <- do.call(rbind, all_data)
combined$position_mb <- combined$position / 1e6
combined$is_recombinant <- as.logical(combined$is_recombinant)
combined$species <- factor(combined$species, levels = labels, labels = strip_labels)

# Per-species oriC/terC for vertical reference lines
markers <- do.call(rbind, lapply(seq_along(species), function(i) {
  sp <- species[i]
  d <- all_data[[sp]]
  if (is.null(d)) return(NULL)
  ori_idx <- which.min(d$cumulative)
  ter_idx <- which.max(d$cumulative)
  data.frame(
    species = labels[i],
    oriC_mb = d$position[ori_idx] / 1e6,
    terC_mb = d$position[ter_idx] / 1e6,
    stringsAsFactors = FALSE
  )
}))
markers$species <- factor(markers$species, levels = labels, labels = strip_labels)

# Build segment data for the cumulative skew curve, coloured by recombinant flag
make_segments <- function(d) {
  if (nrow(d) < 2) return(NULL)
  starts <- d[-nrow(d), ]
  ends <- d[-1, ]
  data.frame(
    species = starts$species,
    x = starts$position_mb,
    xend = ends$position_mb,
    y = starts$cumulative,
    yend = ends$cumulative,
    is_recombinant = starts$is_recombinant | ends$is_recombinant,
    stringsAsFactors = FALSE
  )
}

segments_df <- do.call(rbind, lapply(split(combined, combined$species), make_segments))
segments_df$species <- factor(segments_df$species, levels = strip_labels)
segments_df$status <- ifelse(segments_df$is_recombinant, "Recombinant", "Non-recombinant")

# Plot
p <- ggplot() +
  geom_segment(
    data = segments_df,
    aes(x = x, xend = xend, y = y, yend = yend, colour = status),
    linewidth = 1.0,
    lineend = "round"
  ) +
  geom_vline(
    data = markers,
    aes(xintercept = oriC_mb, linetype = "oriC"),
    colour = "#0F766E", linewidth = 0.6
  ) +
  geom_vline(
    data = markers,
    aes(xintercept = terC_mb, linetype = "terC"),
    colour = "#7C3AED", linewidth = 0.6
  ) +
  facet_wrap(~ species, ncol = 4, scales = "free", axes = "all") +
  scale_colour_manual(
    values = c("Non-recombinant" = "#1E40AF", "Recombinant" = "#DC2626"),
    name = "Region"
  ) +
  scale_linetype_manual(
    values = c("oriC" = "dashed", "terC" = "dashed"),
    name = "Marker",
    guide = guide_legend(override.aes = list(
      colour = c("#0F766E", "#7C3AED"),
      linewidth = 0.6
    ))
  ) +
  labs(
    x = "Genome position (Mb)",
    y = "Cumulative GC skew"
  ) +
  theme_classic(base_size = 11) +
  theme(
    axis.text = element_text(size = 8, colour = "grey25"),
    axis.title = element_text(size = 12),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.line = element_line(colour = "grey30", linewidth = 0.4),
    axis.ticks = element_line(colour = "grey30", linewidth = 0.3),
    panel.spacing.x = unit(1.4, "lines"),
    panel.spacing.y = unit(1.2, "lines"),
    strip.text = element_text(size = 11, face = "italic", margin = margin(b = 5, t = 2)),
    strip.background = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.6, "cm"),
    legend.spacing.x = unit(0.4, "cm"),
    legend.box = "horizontal",
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )

out_path <- file.path(out_dir, "gc_skew_grid_recombinant_full_contigs.png")
ggsave(out_path, p, width = 16, height = 12, dpi = 400, bg = "white")

cat("Saved: ", out_path, "\n", sep = "")