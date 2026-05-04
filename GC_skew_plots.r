library(ggplot2)
library(ggtext)

# Set up paths
base_dir <- 'C:/Users/willi/OneDrive/Documents/University/Year 3/Semester 2/Capstone/Designated Species'
out_dir <- file.path(base_dir, 'Final Figures')

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

species <- c(
  'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
  'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
  'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
  'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
  'V_cholerae', 'S_typhimurium_2'
)

labels <- c(
  'A. baumannii', 'B. pertussis', 'C. jejuni', 'C. difficile', 'E. coli',
  'H. influenzae', 'L. monocytogenes', 'M. tuberculosis',
  'N. gonorrhoeae', 'N. meningitidis', 'P. aeruginosa', 'S. aureus',
  'S. epidermidis', 'S. agalactiae', 'S. pneumoniae', 'S. pyogenes',
  'V. cholerae', 'S. typhimurium'
)

# Read leading and lagging singleton counts
get_lead_pct <- function(folder) {
  lead_path <- file.path(base_dir, folder, 'results', 'spectrum_leading_template.csv')
  lag_path <- file.path(base_dir, folder, 'results', 'spectrum_lagging_template.csv')
  
  if (!file.exists(lead_path) || !file.exists(lag_path)) return(NA)
  
  lead_m <- read.csv(lead_path, row.names = 1)
  lag_m <- read.csv(lag_path, row.names = 1)
  
  n_lead <- sum(lead_m, na.rm = TRUE)
  n_lag <- sum(lag_m, na.rm = TRUE)
  
  if ((n_lead + n_lag) == 0) return(NA)
  
  round(n_lead / (n_lead + n_lag) * 100, 0)
}

cat('Computing leading/lagging singleton proportions per species:\n')
lead_pct <- sapply(species, get_lead_pct)

# Load per-species GC skew window data
all_data <- list()

for (i in seq_along(species)) {
  sp <- species[i]
  path <- file.path(base_dir, sp, 'results', 'gc_skew_windows.csv')
  
  if (!file.exists(path)) {
    cat('Missing:', sp, '\n')
    next
  }
  
  d <- read.csv(path)
  d$species_folder <- sp
  d$species <- labels[i]
  all_data[[sp]] <- d
}

# Recombinant percentage per species
recomb_pct <- sapply(all_data, function(d) {
  round(mean(as.logical(d$is_recombinant), na.rm = TRUE) * 100, 1)
})

# Recombinant strand distribution per species
cat('\nComputing recombinant strand distribution:\n')

recomb_lead_pct <- sapply(all_data, function(d) {
  d$is_recombinant <- as.logical(d$is_recombinant)
  recomb_rows <- d[d$is_recombinant == TRUE, ]
  
  if (nrow(recomb_rows) == 0) return(NA)
  
  n_lead <- sum(recomb_rows$smoothed_skew > 0, na.rm = TRUE)
  n_lag <- sum(recomb_rows$smoothed_skew < 0, na.rm = TRUE)
  
  if ((n_lead + n_lag) == 0) return(NA)
  
  round(n_lead / (n_lead + n_lag) * 100, 0)
})

for (i in seq_along(species)) {
  cat(
    species[i],
    '- singleton lead:', lead_pct[i],
    '%, recomb lead:', recomb_lead_pct[i],
    '%\n'
  )
}


strip_labels <- character(length(labels))

for (i in seq_along(labels)) {
  sp <- species[i]
  rp <- recomb_pct[sp]
  lp <- lead_pct[sp]
  rlp <- recomb_lead_pct[sp]
  
  rp_str <- if (is.na(rp)) 'NA' else paste0(rp, '%')
  
  # First line: species + global recombinant percentage
  line1 <- paste0(
    '<b><i>', labels[i], '</i></b> ',
    '(<b>', rp_str, '</b>)'
  )
  
  # Second line: leading/lagging splits
  parts <- c()
  
  if (!is.na(lp)) {
    parts <- c(
      parts,
      paste0(
        'Sing: <b>', lp, '%</b> Ld / ',
        '<b>', 100 - lp, '%</b> Lg'
      )
    )
  }
  
  if (!is.na(rlp)) {
    parts <- c(
      parts,
      paste0(
        'Rec: <b>', rlp, '%</b> Ld / ',
        '<b>', 100 - rlp, '%</b> Lg'
      )
    )
  }
  
  line2 <- paste(parts, collapse = ' | ')
  
  strip_labels[i] <- paste0(
    line1,
    '<br>',
    "<span style='font-size:9pt;'>", line2, '</span>'
  )
}

names(strip_labels) <- labels

combined <- do.call(rbind, all_data)
combined$position_mb <- combined$position / 1e6
combined$is_recombinant <- as.logical(combined$is_recombinant)
combined$species <- factor(
  combined$species,
  levels = labels,
  labels = strip_labels
)

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

markers$species <- factor(
  markers$species,
  levels = labels,
  labels = strip_labels
)

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

segments_df <- do.call(
  rbind,
  lapply(split(combined, combined$species), make_segments)
)

segments_df$species <- factor(
  segments_df$species,
  levels = strip_labels
)

segments_df$status <- ifelse(
  segments_df$is_recombinant,
  'Recombinant',
  'Non-recombinant'
)

p <- ggplot() +
  geom_segment(
    data = segments_df,
    aes(
      x = x,
      xend = xend,
      y = y,
      yend = yend,
      colour = status
    ),
    linewidth = 1.0,
    lineend = 'round'
  ) +
  geom_vline(
    data = markers,
    aes(xintercept = oriC_mb, linetype = 'oriC'),
    colour = '#0F766E',
    linewidth = 0.6
  ) +
  geom_vline(
    data = markers,
    aes(xintercept = terC_mb, linetype = 'terC'),
    colour = '#7C3AED',
    linewidth = 0.6
  ) +
  facet_wrap(
    ~ species,
    ncol = 4,
    scales = 'free',
    axes = 'all'
  ) +
  scale_colour_manual(
    values = c(
      'Non-recombinant' = '#1E40AF',
      'Recombinant' = '#DC2626'
    ),
    name = 'Region'
  ) +
  scale_linetype_manual(
    values = c(
      'oriC' = 'dashed',
      'terC' = 'dashed'
    ),
    name = 'Marker',
    guide = guide_legend(
      override.aes = list(
        colour = c('#0F766E', '#7C3AED'),
        linewidth = 0.6
      )
    )
  ) +
  labs(
    x = 'Genome position (Mb)',
    y = 'Cumulative GC skew'
  ) +
  theme_classic(base_size = 11) +
  theme(
    axis.text = element_text(size = 8, colour = 'grey25'),
    axis.title = element_text(size = 12),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.line = element_line(colour = 'grey30', linewidth = 0.4),
    axis.ticks = element_line(colour = 'grey30', linewidth = 0.3),
    
    panel.spacing.x = unit(1.4, 'lines'),
    panel.spacing.y = unit(1.6, 'lines'),
    
    strip.text = ggtext::element_markdown(
      size = 11,
      margin = margin(b = 6, t = 4),
      lineheight = 1.2
    ),
    strip.background = element_blank(),
    
    legend.position = 'bottom',
    legend.title = element_text(size = 11, face = 'bold'),
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.6, 'cm'),
    legend.spacing.x = unit(0.4, 'cm'),
    legend.box = 'horizontal',
    
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )

out_path <- file.path(out_dir, 'gc_skew_grid_recombinant_with_strand.png')

ggsave(
  out_path,
  p,
  width = 18,
  height = 13,
  dpi = 400,
  bg = 'white'
)

cat('\nSaved: ', out_path, '\n', sep = '')
