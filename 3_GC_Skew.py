'''
========================================================================
GC skew analysis and replication-context classification pipeline
========================================================================

Overview
--------
This script characterises the GC skew profile of each bacterial
reference genome and classifies each non-recombinant singleton SNV by
its local replication context. The output supports downstream analyses
including the GC skew grid plot (with recombinant overlays), strand-
flipped leading/lagging spectrum analyses, codon-level analyses
restricted to leading or lagging strand contexts, dN/dS analyses
stratified by strand, and proximity-to-origin analyses.

GC skew is calculated as (G - C) / (G + C) in sliding windows along
the genome. The cumulative GC skew is used to identify the origin of
replication (oriC) at the global minimum and the terminus (terC) at
the global maximum. For each individual genomic position, the sign
of the local skew (computed within a larger smoothing window) is used
to classify the position as falling within a leading-strand context
(reference equals leading template, positive skew) or a lagging-
strand context (reference equals lagging template, negative skew).

Input
-----
For each of the 18 species included in this analysis, the script
expects:
  - {raw_data}/{species}_fasta.fna : the reference genome sequence in
    FASTA format.
  - {results}/intermediate_data.pkl : the upstream pickle, used to
    obtain per-sample recombinant SNV positions and singleton details.

Method
------
For each species:

  (1) GC skew profile. The longest contig is taken as the reference
      sequence. GC skew (G - C) / (G + C) is computed in 10 kb
      sliding windows with a 1 kb step. The cumulative skew is then
      taken; oriC is the genomic position at the global minimum and
      terC is the genomic position at the global maximum.

  (2) Recombinant span reconstruction. Per-sample flagged SNV
      positions from the upstream pipeline are merged into contiguous
      spans using the same 2,000 bp merging distance. Each genomic
      position is then flagged as either recombinant or non-
      recombinant. Loose contig name matching is applied to handle
      Parsnp's NZ_ prefix convention.

  (3) Per-position strand classification. A second smoothing pass is
      applied with a 50 kb window and 1 kb step to obtain a smoothed
      local skew value for every genomic position. Positions with
      positive smoothed skew are classified as leading-strand
      contexts; those with negative skew are classified as lagging-
      strand contexts.

  (4) Singleton classification. Each non-recombinant singleton from
      the upstream pipeline is annotated with its local strand
      classification, its distance to the oriC (in base pairs along
      the genome, accounting for circularity), and the strand-flipped
      mutation type (i.e. the mutation expressed on the leading
      strand, by complementing mutations on lagging-strand positions).

  (5) Replication-strand spectra. Singletons are aggregated into per-
      species 4x4 mutation matrices for leading-strand and lagging-
      strand contexts using the strand-normalised mutation type. The
      non-recombinant base composition is also partitioned into
      leading and lagging buckets (with lagging-strand bases
      complemented to express on the leading strand) so the spectra
      can be standardised against the corresponding opportunity.

'''

import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import pickle
import os

base_dir = r'C:\Users\willi\OneDrive\Documents\University\Year 3\Semester 2\Capstone\Designated Species'

species = [
    'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
    'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
    'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
    'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
    'V_cholerae', 'S_typhimurium_2',
]

bases = ['A', 'C', 'G', 'T']

# Sliding window parameters for raw skew profile
skew_window = 10000
skew_step = 1000

# Larger smoothing window for per-position classification
smooth_window = 50000
smooth_step = 1000

# Recombination merging distance (matches upstream pipeline)
recomb_window_length = 2000

def complement_base(b):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return comp.get(b, b)

def complement_mutation(mt):
    ref, alt = mt.split('>')
    return complement_base(ref) + '>' + complement_base(alt)

# Merge per-sample flagged positions into contiguous spans for the chosen contig
def merge_recombinant_spans(suspicious_snvs, window_length, target_contig):
    all_positions = []
    for sample in suspicious_snvs:
        for chrom, pos in suspicious_snvs[sample]:
            if chrom == target_contig or target_contig in chrom or chrom in target_contig:
                all_positions.append(pos)
    if len(all_positions) == 0:
        return []
    all_positions = sorted(set(all_positions))
    spans = []
    region_start = all_positions[0]
    region_end = all_positions[0]
    for p in all_positions[1:]:
        if p - region_end <= window_length:
            region_end = p
        else:
            spans.append((region_start, region_end))
            region_start = p
            region_end = p
    spans.append((region_start, region_end))
    return spans

# Compute raw skew along the genome in sliding windows
def compute_skew_profile(seq, window_size, step):
    positions = []
    skew_values = []
    for start in range(0, len(seq) - window_size + 1, step):
        window = seq[start:start + window_size]
        g = window.count('G')
        c = window.count('C')
        skew = (g - c) / (g + c) if (g + c) > 0 else 0
        positions.append(start + window_size // 2)
        skew_values.append(skew)
    return np.array(positions), np.array(skew_values)

# Distance between two positions on a circular genome
def circular_distance(pos1, pos2, genome_size):
    d = abs(pos1 - pos2)
    return min(d, genome_size - d)

# Build a 4x4 matrix from a Counter of mutation types
def build_matrix(counts_dict):
    matrix = []
    for ref in bases:
        row = [counts_dict.get(ref + '>' + alt, 0) for alt in bases]
        matrix.append(row)
    df = pd.DataFrame(matrix, index=bases, columns=bases)
    df.index.name = 'FROM'
    df.columns.name = 'TO'
    return df

# Standardise a 4x4 spectrum by base counts -> per million and percentage
def standardise_spectrum(matrix, base_counts):
    df_std = matrix.copy().astype(float)
    for b in bases:
        if base_counts.get(b, 0) > 0:
            df_std.loc[b] = df_std.loc[b] / base_counts[b]
    df_per_million = df_std * 1000000
    total_pm = df_per_million.values.sum()
    if total_pm > 0:
        df_pct = (df_per_million / total_pm) * 100
    else:
        df_pct = df_per_million.copy()
    return df_per_million, df_pct

# Count A/C/G/T at non-recombinant positions stratified by local replication strand
# For lagging-strand positions, the base is complemented to express on the leading strand
def count_bases_by_replication_strand(seq, recomb_positions_set, smooth_positions, smooth_values):
    leading_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    lagging_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    
    for i in range(len(seq)):
        pos = i + 1
        if pos in recomb_positions_set:
            continue
        b = seq[i]
        if b not in leading_counts:
            continue
        
        idx = np.argmin(np.abs(smooth_positions - pos))
        local_skew = smooth_values[idx]
        
        if local_skew > 0:
            leading_counts[b] += 1
        else:
            lagging_counts[complement_base(b)] += 1
    
    return leading_counts, lagging_counts

summary_rows = []

for sp in species:
    print('\n' + '=' * 60)
    print('Processing: ' + sp)
    
    species_dir = os.path.join(base_dir, sp)
    raw_data_dir = os.path.join(species_dir, 'raw_data')
    results_dir = os.path.join(species_dir, 'results')
    
    pickle_path = os.path.join(results_dir, 'intermediate_data.pkl')
    
    # FASTA path - try .fna first, fall back to .fasta
    fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fna')
    if not os.path.exists(fasta_path):
        fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fasta')
    
    if not os.path.exists(fasta_path):
        print('  Skipped: no FASTA')
        continue
    if not os.path.exists(pickle_path):
        print('  Skipped: no pickle file')
        continue
    
    # Load FASTA - longest contig only
    seqs = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seqs.append((record.id, str(record.seq).upper()))
    seqs.sort(key=lambda x: -len(x[1]))
    contig_id = seqs[0][0]
    seq = seqs[0][1]
    genome_size = len(seq)
    
    print('Contig: ' + contig_id + ' (' + str(genome_size) + ' bp)')
    
    # Raw skew profile (10 kb window)
    positions, skew_values = compute_skew_profile(seq, skew_window, skew_step)
    cumulative = np.cumsum(skew_values)
    
    oriC_pos = int(positions[np.argmin(cumulative)])
    terC_pos = int(positions[np.argmax(cumulative)])
    
    print('oriC: ' + str(oriC_pos))
    print('terC: ' + str(terC_pos))
    
    # Smoothed skew profile (50 kb window) for per-position classification
    smooth_positions, smooth_values = compute_skew_profile(seq, smooth_window, smooth_step)
    
    # Build smoothed lookup at each window centre
    smoothed_at_window = np.zeros(len(positions))
    for i, p in enumerate(positions):
        idx = np.argmin(np.abs(smooth_positions - p))
        smoothed_at_window[i] = smooth_values[idx]
    
    # Recombinant span reconstruction
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    spans = merge_recombinant_spans(data.get('suspicious_snvs', {}),
                                    recomb_window_length, contig_id)
    
    recomb_positions_set = set()
    for start, end in spans:
        for p in range(start, end + 1):
            recomb_positions_set.add(p)
    recomb_bp = len(recomb_positions_set)
    
    print('Recombinant spans: ' + str(len(spans))
          + ' (' + str(recomb_bp) + ' bp, '
          + str(round(recomb_bp / genome_size * 100, 2)) + '%)')
    
    is_recomb_window = np.array([p in recomb_positions_set for p in positions])
    
    # Save per-window data for the R plotting script
    skew_df = pd.DataFrame({
        'position': positions,
        'gc_skew': skew_values,
        'cumulative': cumulative,
        'smoothed_skew': smoothed_at_window,
        'is_recombinant': is_recomb_window.astype(int),
    })
    skew_df.to_csv(os.path.join(results_dir, 'gc_skew_windows.csv'), index=False)
    
    # Replication-strand-stratified non-recombinant base counts
    leading_base_counts, lagging_base_counts = count_bases_by_replication_strand(
        seq, recomb_positions_set, smooth_positions, smooth_values)
    print('Leading-strand base counts (non-recomb): ' + str(leading_base_counts))
    print('Lagging-strand base counts (non-recomb, expressed on leading): '
          + str(lagging_base_counts))
    
    # Singleton classification using per-position smoothed skew
    singleton_details = data.get('singleton_details', [])
    suspicious_snvs_raw = data.get('suspicious_snvs', {})
    suspicious_snvs = {s: set(tuple(p) for p in pos_list)
                       for s, pos_list in suspicious_snvs_raw.items()}
    
    classified_singletons = []
    n_leading = 0
    n_lagging = 0
    
    leading_mutation_counts = Counter()
    lagging_mutation_counts = Counter()
    
    for s in singleton_details:
        chrom = s['chrom']
        pos = s['pos']
        sample = s['sample']
        mt = s['mutation_type']
        
        if not (chrom == contig_id or contig_id in chrom or chrom in contig_id):
            continue
        
        if sample in suspicious_snvs and (chrom, pos) in suspicious_snvs[sample]:
            continue
        
        idx = np.argmin(np.abs(smooth_positions - pos))
        local_skew = smooth_values[idx]
        local_strand = 'leading' if local_skew > 0 else 'lagging'
        
        if local_strand == 'lagging':
            leading_mt = complement_mutation(mt)
        else:
            leading_mt = mt
        
        dist_to_oriC = circular_distance(pos, oriC_pos, genome_size)
        
        if local_strand == 'leading':
            n_leading += 1
            leading_mutation_counts[leading_mt] += 1
        else:
            n_lagging += 1
            lagging_mutation_counts[leading_mt] += 1
        
        classified_singletons.append({
            'chrom': chrom,
            'pos': pos,
            'sample': sample,
            'majority_base': s.get('majority_base', ''),
            'singleton_base': s.get('singleton_base', ''),
            'mutation_type': mt,
            'local_strand': local_strand,
            'local_skew': round(local_skew, 4),
            'leading_strand_mutation': leading_mt,
            'distance_to_oriC': dist_to_oriC,
        })
    
    classified_df = pd.DataFrame(classified_singletons)
    classified_df.to_csv(os.path.join(results_dir, 'singleton_with_strand.csv'),
                         index=False)
    
    print('Singletons classified: ' + str(len(classified_singletons))
          + ' (leading: ' + str(n_leading)
          + ', lagging: ' + str(n_lagging) + ')')
    
    # Replication-strand spectra (raw, per million, percentage)
    leading_matrix = build_matrix(leading_mutation_counts)
    lagging_matrix = build_matrix(lagging_mutation_counts)
    leading_matrix.to_csv(os.path.join(results_dir, 'spectrum_leading_template.csv'))
    lagging_matrix.to_csv(os.path.join(results_dir, 'spectrum_lagging_template.csv'))
    
    if leading_matrix.values.sum() > 0:
        lead_pm, lead_pct = standardise_spectrum(leading_matrix, leading_base_counts)
        lead_pm.round(2).to_csv(os.path.join(results_dir, 'spectrum_leading_template_per_million.csv'))
        lead_pct.round(2).to_csv(os.path.join(results_dir, 'spectrum_leading_template_percentage.csv'))
    
    if lagging_matrix.values.sum() > 0:
        lag_pm, lag_pct = standardise_spectrum(lagging_matrix, lagging_base_counts)
        lag_pm.round(2).to_csv(os.path.join(results_dir, 'spectrum_lagging_template_per_million.csv'))
        lag_pct.round(2).to_csv(os.path.join(results_dir, 'spectrum_lagging_template_percentage.csv'))
    
    # Per-species metadata text file
    with open(os.path.join(results_dir, 'gc_skew_metadata.txt'), 'w') as f:
        f.write('Contig: ' + contig_id + '\n')
        f.write('Genome size: ' + str(genome_size) + ' bp\n')
        f.write('Skew window: ' + str(skew_window) + ' bp (step ' + str(skew_step) + ')\n')
        f.write('Smoothing window: ' + str(smooth_window) + ' bp (step ' + str(smooth_step) + ')\n')
        f.write('\noriC position: ' + str(oriC_pos) + '\n')
        f.write('terC position: ' + str(terC_pos) + '\n')
        f.write('Cumulative skew range: ' + str(round(cumulative.max() - cumulative.min(), 2)) + '\n')
        f.write('\nRecombinant spans: ' + str(len(spans)) + '\n')
        f.write('Recombinant territory: ' + str(recomb_bp) + ' bp\n')
        f.write('Recombinant percentage: '
                + str(round(recomb_bp / genome_size * 100, 2)) + '%\n')
        f.write('\nClassified singletons: ' + str(len(classified_singletons)) + '\n')
        f.write('Leading-strand singletons: ' + str(n_leading) + '\n')
        f.write('Lagging-strand singletons: ' + str(n_lagging) + '\n')
        f.write('\nLeading non-recomb base counts: ' + str(leading_base_counts) + '\n')
        f.write('Lagging non-recomb base counts: ' + str(lagging_base_counts) + '\n')
    
    # Per-species pickle for downstream
    with open(os.path.join(results_dir, 'gc_skew_data.pkl'), 'wb') as f:
        pickle.dump({
            'contig_id': contig_id,
            'genome_size': genome_size,
            'positions': positions,
            'gc_skew': skew_values,
            'cumulative': cumulative,
            'smooth_positions': smooth_positions,
            'smooth_values': smooth_values,
            'oriC': oriC_pos,
            'terC': terC_pos,
            'spans': spans,
            'is_recomb_window': is_recomb_window,
            'leading_base_counts': leading_base_counts,
            'lagging_base_counts': lagging_base_counts,
            'leading_mutation_counts': dict(leading_mutation_counts),
            'lagging_mutation_counts': dict(lagging_mutation_counts),
        }, f)
    
    summary_rows.append({
        'species': sp,
        'contig': contig_id,
        'genome_size': genome_size,
        'oriC': oriC_pos,
        'terC': terC_pos,
        'recomb_territory_bp': recomb_bp,
        'recomb_territory_pct': round(recomb_bp / genome_size * 100, 2),
        'n_singletons_classified': len(classified_singletons),
        'n_leading': n_leading,
        'n_lagging': n_lagging,
        'leading_fraction': round(n_leading / len(classified_singletons), 3)
                            if len(classified_singletons) > 0 else None,
    })
    
    print('\nSaved:')
    print('  : gc_skew_windows.csv')
    print('  : singleton_with_strand.csv')
    print('  : spectrum_leading_template.csv')
    print('  : spectrum_leading_template_per_million.csv')
    print('  : spectrum_leading_template_percentage.csv')
    print('  : spectrum_lagging_template.csv')
    print('  : spectrum_lagging_template_per_million.csv')
    print('  : spectrum_lagging_template_percentage.csv')
    print('  : gc_skew_metadata.txt')
    print('  : gc_skew_data.pkl')

# Cross-species summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(base_dir, 'gc_skew_summary.csv'), index=False)

print('\n' + '=' * 60)
print('Cross-species summary saved to: ' + os.path.join(base_dir, 'gc_skew_summary.csv'))
print(summary_df.to_string(index=False))