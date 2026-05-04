# 12-mutation spectrum: Streptococcus pneumoniae singletons and super-singletons

library(ggplot2)

# Set up paths
base_dir <- 'C:/Users/willi/OneDrive/Documents/University/Year 3/Semester 2/Capstone/Designated Species'
out_dir <- file.path(base_dir, 'Final Figures')

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Read the master CSV
df <- read.csv(file.path(base_dir, 'master_species_data.csv'))

# The 12 directional mutation types
bases <- c('A', 'C', 'G', 'T')
mut_types <- c('A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T',
               'G>A', 'G>C', 'G>T', 'T>A', 'T>C', 'T>G')

# Find the row for S. pneumoniae - matching on the Species name
sp_row <- df[grepl('pneumoniae', df$Species, ignore.case = TRUE), ]

if (nrow(sp_row) == 0) {
  stop('S. pneumoniae not found in master CSV')
}

# Take the first matching row
sp_row <- sp_row[1, ]

# Build long-format data frame: mutation_type, source, pct
rows <- list()

for (mt in mut_types) {
  ref <- substr(mt, 1, 1)
  alt <- substr(mt, 3, 3)
  
  # Singleton percentage column
  sing_col <- paste0('singleton_pct_', ref, '_to_', alt)
  sing_pct <- as.numeric(sp_row[[sing_col]])
  if (is.na(sing_pct)) sing_pct <- 0
  
  # Super-singleton percentage column
  super_col <- paste0('super_singleton_pct_', ref, '_to_', alt)
  super_pct <- as.numeric(sp_row[[super_col]])
  if (is.na(super_pct)) super_pct <- 0
  
  rows[[length(rows) + 1]] <- data.frame(
    mutation_type = mt,
    source = 'Singletons',
    pct = sing_pct,
    stringsAsFactors = FALSE
  )
  
  rows[[length(rows) + 1]] <- data.frame(
    mutation_type = mt,
    source = 'Super-singletons',
    pct = super_pct,
    stringsAsFactors = FALSE
  )
}

plot_df <- do.call(rbind, rows)
plot_df$mutation_type <- factor(plot_df$mutation_type, levels = mut_types)
plot_df$source <- factor(plot_df$source, levels = c('Singletons', 'Super-singletons'))


source_palette <- c('Singletons' = '#1E40AF', 'Super-singletons' = '#DC2626')

# 4x3 facet grid - one panel per mutation type, two bars per panel
p <- ggplot(plot_df, aes(x = source, y = pct, fill = source)) +
  geom_col(width = 0.7, colour = 'grey25', linewidth = 0.18) +
  facet_wrap(~ mutation_type, ncol = 4, axes = 'all') +
  scale_y_continuous(
    name = 'Mutations (% of total)',
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_x_discrete(name = NULL) +
  scale_fill_manual(values = source_palette, name = NULL) +
  theme_classic(base_size = 11) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 9, colour = 'grey25', angle = 90, hjust = 0.5),
    axis.title.y = element_text(size = 12, margin = margin(r = 8)),
    axis.line = element_line(colour = 'grey30', linewidth = 0.4),
    axis.ticks.y = element_line(colour = 'grey30', linewidth = 0.3),
    panel.spacing.x = unit(1.2, 'lines'),
    panel.spacing.y = unit(1, 'lines'),
    strip.text = element_text(size = 11, face = 'bold', margin = margin(b = 5, t = 2)),
    strip.background = element_blank(),
    legend.position = 'bottom',
    legend.text = element_text(size = 11),
    legend.key.size = unit(0.5, 'cm'),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )


# Save output
out_path <- file.path(out_dir, 'spneumoniae_singleton_vs_supersingleton_grid.png')
ggsave(out_path, p, width = 12, height = 8, dpi = 400, bg = 'white')

cat('Saved: ', out_path, '\n', sep = '')