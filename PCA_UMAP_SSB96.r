library(ggplot2)
library(ggrepel)
library(uwot)

base_dir <- r'C:\Users\[USER]\...\Species_folder'
out_dir <- file.path(base_dir, "Final Figures")

df <- read.csv(file.path(base_dir, "master_species_data.csv"))

species_labels <- sapply(strsplit(df$Species, "_"), function(x) {
  if (length(x) >= 2) {
    paste0(substr(x[1], 1, 1), ". ", paste(x[-1], collapse = " "))
  } else {
    x[1]
  }
})

sbs96_pct_cols <- grep("^sbs96_pct_", names(df), value = TRUE)
cat("SBS-96 percentage columns:", length(sbs96_pct_cols), "\n")

X <- df[, sbs96_pct_cols]
complete <- complete.cases(X) & rowSums(X) > 0
X <- X[complete, ]
meta <- df[complete, ]
labels <- species_labels[complete]

cat("Species with complete data:", nrow(X), "\n")

plot_theme <- function() {
  theme_classic(base_size = 11) +
    theme(
      axis.text = element_text(size = 9, colour = "grey25"),
      axis.title = element_text(size = 12, margin = margin(r = 8)),
      axis.line = element_line(colour = "grey30", linewidth = 0.4),
      axis.ticks = element_line(colour = "grey30", linewidth = 0.3),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 9),
      legend.position = "right",
      plot.margin = margin(t = 8, r = 8, b = 8, l = 8)
    )
}

metadata_cols <- intersect(c("Phylum", "Class", "Family", "Gram_stain", "Host_primary_location"),
                           names(meta))

pca <- prcomp(X, center = TRUE, scale. = TRUE)
var_exp <- round(pca$sdev^2 / sum(pca$sdev^2) * 100, 1)
cat("PCA  PC1:", var_exp[1], "% | PC2:", var_exp[2], "%\n")

pca_plot_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  species = labels,
  stringsAsFactors = FALSE
)
for (mc in metadata_cols) {
  pca_plot_df[[mc]] <- meta[[mc]]
}

set.seed(42)
umap_layout <- umap(
  as.matrix(X),
  n_neighbors = min(15, nrow(X) - 1),
  min_dist = 0.3,
  metric = "euclidean"
)

umap_plot_df <- data.frame(
  UMAP1 = umap_layout[, 1],
  UMAP2 = umap_layout[, 2],
  species = labels,
  stringsAsFactors = FALSE
)
for (mc in metadata_cols) {
  umap_plot_df[[mc]] <- meta[[mc]]
}

plot_embedding <- function(plot_df, x_var, y_var, colour_var, x_label, y_label,
                           file_path) {
  plot_df[[colour_var]] <- gsub("_", " ", plot_df[[colour_var]])
  legend_title <- gsub("_", " ", colour_var)
  
  p <- ggplot(plot_df, aes(x = .data[[x_var]], y = .data[[y_var]],
                           colour = .data[[colour_var]])) +
    geom_point(size = 4, alpha = 0.85) +
    geom_text_repel(aes(label = species), size = 3, fontface = "italic",
                    max.overlaps = 30, colour = "grey20",
                    segment.colour = "grey60", segment.size = 0.3) +
    labs(x = x_label, y = y_label, colour = legend_title) +
    plot_theme()
  ggsave(file_path, p, width = 10, height = 7, dpi = 400, bg = "white")
  cat("Saved: ", file_path, "\n", sep = "")
}

for (mc in metadata_cols) {
  plot_embedding(
    pca_plot_df, "PC1", "PC2", mc,
    paste0("PC1 (", var_exp[1], "%)"),
    paste0("PC2 (", var_exp[2], "%)"),
    file.path(out_dir, paste0("pca_sbs96_", mc, ".png"))
  )
}

for (mc in metadata_cols) {
  plot_embedding(
    umap_plot_df, "UMAP1", "UMAP2", mc,
    "UMAP1", "UMAP2",
    file.path(out_dir, paste0("umap_sbs96_", mc, ".png"))
  )
}

loadings <- data.frame(
  variable = rownames(pca$rotation),
  PC1 = pca$rotation[, 1],
  PC2 = pca$rotation[, 2]
)
top_loadings <- loadings[order(loadings$PC1^2 + loadings$PC2^2, decreasing = TRUE)[1:20], ]
top_loadings$variable <- gsub("sbs96_pct_", "", top_loadings$variable)
top_loadings$variable <- gsub("_to_", ">", top_loadings$variable)

p_load <- ggplot(top_loadings, aes(x = PC1, y = PC2, label = variable)) +
  geom_segment(aes(x = 0, y = 0, xend = PC1, yend = PC2),
               arrow = arrow(length = unit(0.18, "cm")), colour = "grey50") +
  geom_text_repel(size = 3) +
  labs(
    x = paste0("PC1 (", var_exp[1], "%)"),
    y = paste0("PC2 (", var_exp[2], "%)")
  ) +
  plot_theme()
ggsave(file.path(out_dir, "pca_sbs96_loadings.png"),
       p_load, width = 9, height = 8, dpi = 400, bg = "white")

cat("\nDone.\n")
