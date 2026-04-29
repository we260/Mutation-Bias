'''
========================================================================
Codon and coding-context singleton analysis pipeline
========================================================================

Overview
--------
This script characterises singleton mutations within their codon and
coding-region context for each bacterial species, separating them by
replication strand (leading versus lagging) where the GC skew profile
permits. The output supports analyses of codon-level mutation bias,
codon-position bias, coding versus intergenic mutation density, and
the ratio of non-synonymous to synonymous substitutions (d_N/d_S),
with all measurements stratified by replication strand context where
appropriate.

Codons are defined by CDS features in the GFF annotation file. For
genes on the minus strand, the coding sequence is taken as the
reverse complement of the reference sequence so that codons are
expressed in the gene's coding direction. Singleton mutations falling
within CDS regions are similarly complemented when the gene is on the
minus strand, ensuring codon membership and codon position are
expressed consistently with respect to the gene's frame.

Input
-----
For each species, the script expects:
  - {raw_data}/{species}_fasta.fna or .fasta : the reference sequence.
  - {raw_data}/genomic.gff or {raw_data}/ncbi_dataset/data/*/genomic.gff
    : the genome annotation in GFF3 format.
  - {results}/intermediate_data.pkl : the upstream pipeline pickle,
    used for non-recombinant base counts and singleton details.
  - {results}/singleton_with_strand.csv : per-singleton classification
    from the GC skew pipeline, providing local strand context.

Method
------
For each species:

  (1) GFF parsing. CDS features on the longest contig are loaded as a
      list of (start, end, strand) tuples. A position-to-strand map is
      built so each genomic position can be looked up as either CDS
      (with strand) or non-CDS (intergenic).

  (2) Coding versus intergenic opportunity. The non-recombinant base
      count is partitioned into a CDS component (counted on the gene's
      coding strand by complementing minus-strand bases) and an
      intergenic component (counted on the reference strand only,
      since intergenic regions have no transcribed strand by
      definition).

  (3) Codon and codon-position annotation. For each singleton in a
      CDS, the codon containing the singleton is determined from the
      gene's start coordinate and reading frame. The codon position
      (1, 2, or 3) and the wild-type and mutant codons are recorded.
      For minus-strand genes, the codon and mutation type are
      reverse-complemented to express them in the coding direction.

  (4) Strand stratification. Each CDS singleton is also tagged with
      the replication strand context (leading or lagging) inherited
      from the upstream gc_skew pipeline. Tallies are produced for
      each combination of codon, codon position, and replication
      strand context.

  (5) Synonymous and non-synonymous classification. Each CDS singleton
      is classified using the standard genetic code: a mutation is
      synonymous if the wild-type and mutant codons translate to the
      same amino acid, non-synonymous otherwise (stop codons are
      counted separately).

  (6) Opportunity for d_N/d_S. For each codon in the genome, all 9
      possible single-base substitutions are evaluated and classified
      as synonymous, non-synonymous, or stop-introducing. Summing
      these across all CDS positions yields the total synonymous and
      non-synonymous opportunities, against which the observed
      counts are normalised. d_N/d_S is then computed as
      (n_observed / n_opportunity) divided by (s_observed /
      s_opportunity).

Output
------
Per species (results directory):
  - codon_singletons.csv : per-singleton table with codon, codon
    position, wild-type and mutant codons, amino acids, syn/non-syn
    classification, gene strand, and replication strand.
  - codon_opportunity.csv : long-format counts of each codon in the
    genome (CDS only), and the synonymous and non-synonymous
    opportunity at each codon position.
  - codon_summary.csv : aggregated counts by codon and replication
    strand context (leading, lagging, total).
  - codon_position_summary.csv : aggregated counts by codon position
    (1, 2, 3) and replication strand.
  - coding_vs_intergenic_summary.csv : per-mutation-type counts and
    opportunities for CDS and intergenic regions.
  - dnds_summary.txt : the d_N/d_S calculation including the
    intermediate counts.
  - codon_data.pkl : pickled dictionary of all numerical arrays for
    downstream use.

A combined cross-species summary is written to the parent directory as
codon_summary_all.csv.
========================================================================
'''

import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
import os
import glob
from collections import Counter, defaultdict

base_dir = r'C:\Users\willi\OneDrive\Documents\University\Year 3\Semester 2\Capstone\Designated Species'

species = [
    'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
    'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
    'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
    'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
    'V_cholerae', 'S_typhimurium_2',
]

bases = ['A', 'C', 'G', 'T']

# Standard genetic code
genetic_code = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def find_fasta(folder, species_name):
    for ext in ['.fna', '.fasta']:
        candidate = os.path.join(folder, species_name + '_fasta' + ext)
        if os.path.exists(candidate):
            return candidate
    matches = []
    for ext in ['*.fna', '*.fasta']:
        matches.extend(glob.glob(os.path.join(folder, ext)))
    if matches:
        return matches[0]
    return None

def find_gff(folder):
    candidates = glob.glob(os.path.join(folder, '*.gff'))
    if candidates:
        return candidates[0]
    nested = glob.glob(os.path.join(folder, 'ncbi_dataset', 'data', '*', '*.gff'))
    if nested:
        return nested[0]
    return None

def reverse_complement(s):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(s))

def complement_base(b):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return comp.get(b, b)

def complement_mutation(mt):
    ref, alt = mt.split('>')
    return complement_base(ref) + '>' + complement_base(alt)

# Parse GFF for CDS features on the chosen contig
def load_cds_features(gff_path, target_contig):
    features = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom = parts[0]
            feature_type = parts[2]
            if feature_type != 'CDS':
                continue
            if not (chrom == target_contig or target_contig in chrom or chrom in target_contig):
                continue
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            features.append((start, end, strand))
    return features

# Build a position lookup: pos -> (gene_start, gene_end, strand) or None
def build_cds_position_map(features):
    cds_map = {}
    for start, end, strand in features:
        for p in range(start, end + 1):
            cds_map[p] = (start, end, strand)
    return cds_map

# For each codon position in the genome, determine syn / non-syn / stop opportunity
def compute_codon_opportunity(seq, features):
    syn_opp = 0
    nonsyn_opp = 0
    stop_opp = 0
    codon_count = Counter()
    
    for start, end, strand in features:
        # Extract gene sequence in coding direction
        gene_seq = seq[start - 1:end]
        if strand == '-':
            gene_seq = reverse_complement(gene_seq)
        
        # Iterate codons
        n_codons = len(gene_seq) // 3
        for i in range(n_codons):
            codon = gene_seq[i * 3:(i + 1) * 3]
            if 'N' in codon or len(codon) != 3:
                continue
            
            wt_aa = genetic_code.get(codon)
            if wt_aa is None:
                continue
            codon_count[codon] += 1
            
            # All 9 possible single-base mutations
            for cp in range(3):
                wt_base = codon[cp]
                for alt in bases:
                    if alt == wt_base:
                        continue
                    mut_codon = codon[:cp] + alt + codon[cp+1:]
                    mut_aa = genetic_code.get(mut_codon)
                    if mut_aa is None:
                        continue
                    if mut_aa == '*':
                        stop_opp += 1
                    elif mut_aa == wt_aa:
                        syn_opp += 1
                    else:
                        nonsyn_opp += 1
    
    return syn_opp, nonsyn_opp, stop_opp, codon_count

# Compute coding strand base counts (complement bases on minus strand)
def count_cds_bases_coding(seq, cds_map, recomb_positions):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for pos, (start, end, strand) in cds_map.items():
        if pos in recomb_positions:
            continue
        if pos < 1 or pos > len(seq):
            continue
        b = seq[pos - 1]
        if strand == '-':
            b = complement_base(b)
        if b in counts:
            counts[b] += 1
    return counts

def count_intergenic_bases(seq, cds_map, recomb_positions):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for i in range(len(seq)):
        pos = i + 1
        if pos in cds_map:
            continue
        if pos in recomb_positions:
            continue
        b = seq[i]
        if b in counts:
            counts[b] += 1
    return counts

# Reconstruct recombinant position set from upstream pickle
def get_recomb_positions(suspicious_snvs, target_contig, window_length=2000):
    all_positions = []
    for sample in suspicious_snvs:
        for chrom, pos in suspicious_snvs[sample]:
            if chrom == target_contig or target_contig in chrom or chrom in target_contig:
                all_positions.append(pos)
    if not all_positions:
        return set()
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
    
    recomb_set = set()
    for start, end in spans:
        for p in range(start, end + 1):
            recomb_set.add(p)
    return recomb_set

cross_species_summary = []

for sp in species:
    print('\n' + '=' * 60)
    print('Processing: ' + sp)
    
    species_dir = os.path.join(base_dir, sp)
    raw_data_dir = os.path.join(species_dir, 'raw_data')
    results_dir = os.path.join(species_dir, 'results')
    
    fasta_path = find_fasta(raw_data_dir, sp)
    gff_path = find_gff(raw_data_dir)
    pickle_path = os.path.join(results_dir, 'intermediate_data.pkl')
    strand_path = os.path.join(results_dir, 'singleton_with_strand.csv')
    
    if fasta_path is None:
        print('  Skipped: no FASTA')
        continue
    if gff_path is None:
        print('  Skipped: no GFF')
        continue
    if not os.path.exists(pickle_path):
        print('  Skipped: no upstream pickle')
        continue
    if not os.path.exists(strand_path):
        print('  Skipped: no singleton_with_strand.csv')
        continue
    
    # Load FASTA - longest contig
    seqs = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seqs.append((record.id, str(record.seq).upper()))
    seqs.sort(key=lambda x: -len(x[1]))
    contig_id = seqs[0][0]
    seq = seqs[0][1]
    genome_size = len(seq)
    
    print('Contig: ' + contig_id + ' (' + str(genome_size) + ' bp)')
    
    # Load CDS features and build position map
    features = load_cds_features(gff_path, contig_id)
    cds_map = build_cds_position_map(features)
    print('CDS features: ' + str(len(features)))
    print('CDS positions: ' + str(len(cds_map)))
    
    if len(features) == 0:
        print('  Skipped: no CDS features matching contig')
        continue
    
    # Load upstream data
    with open(pickle_path, 'rb') as f:
        upstream = pickle.load(f)
    
    suspicious_snvs_raw = upstream.get('suspicious_snvs', {})
    suspicious_snvs = {s: set(tuple(p) for p in pos_list)
                       for s, pos_list in suspicious_snvs_raw.items()}
    
    recomb_positions = get_recomb_positions(suspicious_snvs, contig_id)
    print('Recombinant positions: ' + str(len(recomb_positions)))
    
    # Coding (strand-corrected) and intergenic base counts excluding recombinant
    cds_base_counts = count_cds_bases_coding(seq, cds_map, recomb_positions)
    intergenic_base_counts = count_intergenic_bases(seq, cds_map, recomb_positions)
    print('CDS coding-strand base counts: ' + str(cds_base_counts))
    print('Intergenic base counts: ' + str(intergenic_base_counts))
    
    # d_N/d_S opportunity
    syn_opp, nonsyn_opp, stop_opp, codon_count_genome = compute_codon_opportunity(seq, features)
    print('Synonymous opportunity: ' + str(syn_opp))
    print('Non-synonymous opportunity: ' + str(nonsyn_opp))
    print('Stop-introducing opportunity: ' + str(stop_opp))
    
    # Load singleton-with-strand CSV
    sw = pd.read_csv(strand_path)
    
    # Annotate each singleton with codon info
    codon_singletons = []
    coding_vs_intergenic = {'CDS': defaultdict(int), 'intergenic': defaultdict(int)}
    
    syn_count = 0
    nonsyn_count = 0
    stop_count = 0
    
    for i in range(len(sw)):
        chrom = sw['chrom'].iloc[i]
        pos = int(sw['pos'].iloc[i])
        sample = sw['sample'].iloc[i]
        mt = sw['mutation_type'].iloc[i]
        local_strand = sw['local_strand'].iloc[i]
        
        # Coding/intergenic classification for ALL singletons
        if pos in cds_map:
            coding_vs_intergenic['CDS'][mt] += 1
            
            # Now do codon-level annotation
            gene_start, gene_end, gene_strand = cds_map[pos]
            
            # Position within gene (1-based offset)
            if gene_strand == '+':
                offset = pos - gene_start  # 0-indexed
                gene_seq = seq[gene_start - 1:gene_end]
                # Codon index and position
                codon_idx = offset // 3
                codon_pos = offset % 3 + 1
                wt_codon = gene_seq[codon_idx * 3:codon_idx * 3 + 3]
                # Apply the singleton change
                singleton_base = mt.split('>')[1]
                mut_codon = wt_codon[:codon_pos - 1] + singleton_base + wt_codon[codon_pos:]
                coding_mt = mt
            else:
                # Minus strand - work in coding direction
                offset = gene_end - pos  # 0-indexed from end
                gene_seq = reverse_complement(seq[gene_start - 1:gene_end])
                codon_idx = offset // 3
                codon_pos = offset % 3 + 1
                wt_codon = gene_seq[codon_idx * 3:codon_idx * 3 + 3]
                # Complement the mutation to express in coding direction
                coding_mt = complement_mutation(mt)
                singleton_base = coding_mt.split('>')[1]
                mut_codon = wt_codon[:codon_pos - 1] + singleton_base + wt_codon[codon_pos:]
            
            if 'N' in wt_codon or 'N' in mut_codon or len(wt_codon) != 3:
                continue
            
            wt_aa = genetic_code.get(wt_codon)
            mut_aa = genetic_code.get(mut_codon)
            if wt_aa is None or mut_aa is None:
                continue
            
            if mut_aa == '*':
                effect = 'stop'
                stop_count += 1
            elif mut_aa == wt_aa:
                effect = 'synonymous'
                syn_count += 1
            else:
                effect = 'non-synonymous'
                nonsyn_count += 1
            
            codon_singletons.append({
                'chrom': chrom,
                'pos': pos,
                'sample': sample,
                'mutation_type': mt,
                'coding_mutation_type': coding_mt,
                'gene_strand': gene_strand,
                'gene_start': gene_start,
                'gene_end': gene_end,
                'codon': wt_codon,
                'mutated_codon': mut_codon,
                'codon_position': codon_pos,
                'wild_type_aa': wt_aa,
                'mutant_aa': mut_aa,
                'effect': effect,
                'local_strand': local_strand,
            })
        else:
            coding_vs_intergenic['intergenic'][mt] += 1
    
    print('CDS singletons: ' + str(len(codon_singletons)))
    print('  Synonymous: ' + str(syn_count))
    print('  Non-synonymous: ' + str(nonsyn_count))
    print('  Stop: ' + str(stop_count))
    
    # Save per-singleton table
    codon_df = pd.DataFrame(codon_singletons)
    codon_df.to_csv(os.path.join(results_dir, 'codon_singletons.csv'), index=False)
    
    # Codon-level summary tallies, stratified by replication strand
    codon_summary_rows = []
    for codon in sorted(genetic_code.keys()):
        sub = codon_df[codon_df['codon'] == codon] if len(codon_df) > 0 else codon_df
        n_total = len(sub)
        n_lead = int((sub['local_strand'] == 'leading').sum()) if n_total > 0 else 0
        n_lag = int((sub['local_strand'] == 'lagging').sum()) if n_total > 0 else 0
        codon_summary_rows.append({
            'codon': codon,
            'amino_acid': genetic_code[codon],
            'genome_count': codon_count_genome.get(codon, 0),
            'singleton_count_total': n_total,
            'singleton_count_leading': n_lead,
            'singleton_count_lagging': n_lag,
        })
    codon_summary_df = pd.DataFrame(codon_summary_rows)
    codon_summary_df.to_csv(os.path.join(results_dir, 'codon_summary.csv'), index=False)
    
    # Codon position summary
    codon_pos_rows = []
    for cp in [1, 2, 3]:
        sub = codon_df[codon_df['codon_position'] == cp] if len(codon_df) > 0 else codon_df
        n_total = len(sub)
        n_lead = int((sub['local_strand'] == 'leading').sum()) if n_total > 0 else 0
        n_lag = int((sub['local_strand'] == 'lagging').sum()) if n_total > 0 else 0
        codon_pos_rows.append({
            'codon_position': cp,
            'singleton_count_total': n_total,
            'singleton_count_leading': n_lead,
            'singleton_count_lagging': n_lag,
        })
    codon_pos_df = pd.DataFrame(codon_pos_rows)
    codon_pos_df.to_csv(os.path.join(results_dir, 'codon_position_summary.csv'), index=False)
    
    # Coding vs intergenic summary
    cvi_rows = []
    for ref in bases:
        for alt in bases:
            if ref == alt:
                continue
            mt = ref + '>' + alt
            cds_obs = coding_vs_intergenic['CDS'].get(mt, 0)
            ig_obs = coding_vs_intergenic['intergenic'].get(mt, 0)
            cvi_rows.append({
                'mutation_type': mt,
                'cds_observed': cds_obs,
                'cds_opportunity': cds_base_counts[ref],
                'intergenic_observed': ig_obs,
                'intergenic_opportunity': intergenic_base_counts[ref],
            })
    cvi_df = pd.DataFrame(cvi_rows)
    cvi_df.to_csv(os.path.join(results_dir, 'coding_vs_intergenic_summary.csv'),
                  index=False)
    
    # Codon opportunity table
    codon_opp_rows = []
    for codon in sorted(genetic_code.keys()):
        codon_opp_rows.append({
            'codon': codon,
            'amino_acid': genetic_code[codon],
            'genome_count': codon_count_genome.get(codon, 0),
        })
    codon_opp_df = pd.DataFrame(codon_opp_rows)
    codon_opp_df.to_csv(os.path.join(results_dir, 'codon_opportunity.csv'), index=False)
    
    # d_N/d_S calculation
    if syn_opp > 0 and nonsyn_opp > 0 and syn_count > 0:
        dn = nonsyn_count / nonsyn_opp
        ds = syn_count / syn_opp
        dn_ds = dn / ds if ds > 0 else None
    else:
        dn = None
        ds = None
        dn_ds = None
    
    # d_N/d_S text summary
    with open(os.path.join(results_dir, 'dnds_summary.txt'), 'w') as f:
        f.write('Synonymous opportunity: ' + str(syn_opp) + '\n')
        f.write('Non-synonymous opportunity: ' + str(nonsyn_opp) + '\n')
        f.write('Stop-introducing opportunity: ' + str(stop_opp) + '\n')
        f.write('\nObserved synonymous singletons: ' + str(syn_count) + '\n')
        f.write('Observed non-synonymous singletons: ' + str(nonsyn_count) + '\n')
        f.write('Observed stop-introducing singletons: ' + str(stop_count) + '\n')
        f.write('\ndN: ' + (str(round(dn, 6)) if dn is not None else 'NA') + '\n')
        f.write('dS: ' + (str(round(ds, 6)) if ds is not None else 'NA') + '\n')
        f.write('dN/dS: ' + (str(round(dn_ds, 4)) if dn_ds is not None else 'NA') + '\n')
    
    # Pickle for downstream
    with open(os.path.join(results_dir, 'codon_data.pkl'), 'wb') as f:
        pickle.dump({
            'contig_id': contig_id,
            'genome_size': genome_size,
            'cds_features': features,
            'cds_base_counts': cds_base_counts,
            'intergenic_base_counts': intergenic_base_counts,
            'syn_opportunity': syn_opp,
            'nonsyn_opportunity': nonsyn_opp,
            'stop_opportunity': stop_opp,
            'syn_count': syn_count,
            'nonsyn_count': nonsyn_count,
            'stop_count': stop_count,
            'codon_count_genome': dict(codon_count_genome),
            'dnds': dn_ds,
            'dn': dn,
            'ds': ds,
        }, f)
    
    cross_species_summary.append({
        'species': sp,
        'genome_size': genome_size,
        'cds_features': len(features),
        'cds_singletons': len(codon_singletons),
        'syn_count': syn_count,
        'nonsyn_count': nonsyn_count,
        'stop_count': stop_count,
        'syn_opportunity': syn_opp,
        'nonsyn_opportunity': nonsyn_opp,
        'dN': round(dn, 6) if dn is not None else None,
        'dS': round(ds, 6) if ds is not None else None,
        'dN_dS': round(dn_ds, 4) if dn_ds is not None else None,
    })
    
    print('\nSaved:')
    print('  : codon_singletons.csv')
    print('  : codon_opportunity.csv')
    print('  : codon_summary.csv')
    print('  : codon_position_summary.csv')
    print('  : coding_vs_intergenic_summary.csv')
    print('  : dnds_summary.txt')
    print('  : codon_data.pkl')

cross_df = pd.DataFrame(cross_species_summary)
cross_df.to_csv(os.path.join(base_dir, 'codon_summary_all.csv'), index=False)

print('\n' + '=' * 60)
print('Saved cross-species summary to: ' + os.path.join(base_dir, 'codon_summary_all.csv'))
print(cross_df.to_string(index=False))