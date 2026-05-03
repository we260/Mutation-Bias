'''
Overview
This script extracts the 96-channel mutational spectrum for each
species in the dataset. Each non-recombinant singleton is
recorded together with its immediate 5' and 3' flanking bases. Where
the singleton's reference base is a purine (A or G), the trinucleotide
context and mutation are reverse-complemented so that all mutations
are expressed in pyrimidine-centric form (i.e. C>X or T>X with the
flanking bases flipped accordingly). This reduces the 192 possible
directional channels into 96 pyrimidine-centric options.
(6 substitution types x 16 trinucleotide contexts).

Input
For each species:
  - {raw_data}/{species}_fasta.fna or .fasta (reference genome).
  - {results}/intermediate_data.pkl : the upstream pickle, containing
    singleton details and per-sample recombinant SNV positions.

Method
For each species:

  (1) Recombinant span reconstruction. The same merging logic as the
      upstream pipeline is applied to obtain a set of all recombinant
      genomic positions on the longest contig (loose contig matching).

  (2) Trinucleotide opportunity. The reference sequence is scanned
      with a 3 bp sliding window (step 1). For each non-recombinant
      trinucleotide where the central base is a pyrimidine (C or T),
      the trinucleotide is recorded directly. For trinucleotides
      where the central base is a purine (A or G), the reverse
      complement is recorded instead. This produces a count for each
      of the 16 pyrimidine-centric trinucleotides in the non-
      recombinant genome.

  (3) Per-singleton context. Each non-recombinant singleton is
      annotated with its 3 bp context (5' base, central base,
      3' base) read from the reference. If the central base is a
      purine, the context and mutation are reverse-complemented to
      pyrimidine form. The resulting 96-channel label takes the form
      "N[X>Y]N", e.g. "A[C>T]G".

  (4) Opportunity normalisation. Singleton counts in each of the 96
      channels are divided by the count of the corresponding 16
      pyrimidine-centric trinucleotides in the non-recombinant genome
      (each context contributes opportunity to three substitution
      types: e.g. ACG provides opportunity for C>A, C>G, and C>T).
      Per-million and percentage forms are produced.

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
pyrimidines = {'C', 'T'}
window_length = 2000

# Six pyrimidine-centric substitution types
pyrimidine_subs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

# 16 pyrimidine-centric trinucleotide contexts (central base = C or T)
def all_pyrimidine_trinucs():
    contexts = []
    for left in bases:
        for centre in ['C', 'T']:
            for right in bases:
                contexts.append(left + centre + right)
    return contexts

trinuc_contexts = all_pyrimidine_trinucs()  # length 32 (16 C-centred, 16 T-centred)

# 96 channels in canonical Alexandrov order
def all_sbs96_channels():
    channels = []
    for sub in pyrimidine_subs:
        ref = sub.split('>')[0]
        alt = sub.split('>')[1]
        for left in bases:
            for right in bases:
                channels.append(left + '[' + sub + ']' + right)
    return channels

sbs96_channels = all_sbs96_channels()  # length 96

def reverse_complement(s):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(s))

def complement_base(b):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return comp.get(b, b)

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

# Convert any (left, ref, alt, right) to its pyrimidine-centric SBS96 channel
def to_sbs96(left, ref, alt, right):
    if ref in pyrimidines:
        return left + '[' + ref + '>' + alt + ']' + right
    else:
        # Purine-centred -> flip to pyrimidine frame
        new_left = complement_base(right)
        new_right = complement_base(left)
        new_ref = complement_base(ref)
        new_alt = complement_base(alt)
        return new_left + '[' + new_ref + '>' + new_alt + ']' + new_right

# Convert any trinucleotide to its pyrimidine-centric form
def to_pyrimidine_trinuc(trinuc):
    if trinuc[1] in pyrimidines:
        return trinuc
    return reverse_complement(trinuc)

# Count pyrimidine-centric trinucleotides at non-recombinant positions
def count_trinuc_opportunity(seq, recomb_positions):
    counts = Counter()
    for i in range(1, len(seq) - 1):
        pos = i + 1  # 1-based central position
        if pos in recomb_positions:
            continue
        trinuc = seq[i - 1:i + 2]
        if 'N' in trinuc or len(trinuc) != 3:
            continue
        canonical = to_pyrimidine_trinuc(trinuc)
        counts[canonical] += 1
    return counts

cross_summary = []

for sp in species:
    print('\n' + '=' * 60)
    print('Processing: ' + sp)
    
    species_dir = os.path.join(base_dir, sp)
    raw_data_dir = os.path.join(species_dir, 'raw_data')
    results_dir = os.path.join(species_dir, 'results')
    pickle_path = os.path.join(results_dir, 'intermediate_data.pkl')
    
    # FASTA
    fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fna')
    if not os.path.exists(fasta_path):
        fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fasta')
    if not os.path.exists(fasta_path):
        print('  Skipped: no FASTA')
        continue
    if not os.path.exists(pickle_path):
        print('  Skipped: no upstream pickle')
        continue
    
    # Load FASTA - longest contig
    seqs = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seqs.append((record.id, str(record.seq).upper()))
    seqs.sort(key=lambda x: -len(x[1]))
    contig_id = seqs[0][0]
    seq = seqs[0][1]
    
    # Load upstream data
    with open(pickle_path, 'rb') as f:
        upstream = pickle.load(f)
    
    suspicious_snvs_raw = upstream.get('suspicious_snvs', {})
    suspicious_snvs = {s: set(tuple(p) for p in pos_list)
                       for s, pos_list in suspicious_snvs_raw.items()}
    
    spans = merge_recombinant_spans(suspicious_snvs, window_length, contig_id)
    recomb_positions = set()
    for start, end in spans:
        for p in range(start, end + 1):
            recomb_positions.add(p)
    print('  Recombinant positions: ' + str(len(recomb_positions)))
    
    # Trinucleotide opportunity (non-recombinant, pyrimidine-centric)
    trinuc_opp = count_trinuc_opportunity(seq, recomb_positions)
    print('  Pyrimidine trinucleotide contexts found: ' + str(len(trinuc_opp)))
    
    # Singleton SBS96 annotation
    singleton_details = upstream.get('singleton_details', [])
    
    sbs96_rows = []
    sbs96_counts = Counter()
    
    for s in singleton_details:
        chrom = s['chrom']
        pos = s['pos']
        sample = s['sample']
        majority_base = s.get('majority_base', '')
        singleton_base = s.get('singleton_base', '')
        
        if not (chrom == contig_id or contig_id in chrom or chrom in contig_id):
            continue
        
        # Skip recombinant singletons (the upstream pipeline already excludes
        # these but this is a safety net)
        if sample in suspicious_snvs and (chrom, pos) in suspicious_snvs[sample]:
            continue
        
        # Get flanking bases from the reference
        if pos < 2 or pos >= len(seq):
            continue
        
        left = seq[pos - 2]
        right = seq[pos]  # pos is 1-based, so seq[pos] is the position after
        # Sanity check: seq[pos - 1] should match majority_base for most singletons
        # but not always (because majority might differ from reference).
        # the singleton's recorded majority_base as the "ref" of the substitution.
        
        if left not in bases or right not in bases:
            continue
        if majority_base not in bases or singleton_base not in bases:
            continue
        
        channel = to_sbs96(left, majority_base, singleton_base, right)
        # Also record the canonical pyrimidine trinucleotide context
        if majority_base in pyrimidines:
            tri_canonical = left + majority_base + right
        else:
            tri_canonical = reverse_complement(left + majority_base + right)
        
        sbs96_counts[channel] += 1
        sbs96_rows.append({
            'chrom': chrom,
            'pos': pos,
            'sample': sample,
            'majority_base': majority_base,
            'singleton_base': singleton_base,
            'left_base': left,
            'right_base': right,
            'pyrimidine_central_base': tri_canonical[1],
            'trinucleotide_context': tri_canonical,
            'sbs96_channel': channel,
        })
    
    print('  Singletons assigned to channels: ' + str(len(sbs96_rows)))
    
    sbs96_df = pd.DataFrame(sbs96_rows)
    sbs96_df.to_csv(os.path.join(results_dir, 'sbs96_singletons.csv'), index=False)
    
    # Long-format spectrum: 96 rows
    spectrum_rows = []
    for ch in sbs96_channels:
        # Recover trinucleotide context from the channel string
        left = ch[0]
        ref = ch[2]
        alt = ch[4]
        right = ch[6]
        tri = left + ref + right
        
        observed = sbs96_counts.get(ch, 0)
        opp = trinuc_opp.get(tri, 0)
        rate_per_million = (observed / opp * 1000000) if opp > 0 else 0
        
        spectrum_rows.append({
            'sbs96_channel': ch,
            'substitution': ref + '>' + alt,
            'left_base': left,
            'right_base': right,
            'trinucleotide_context': tri,
            'observed': observed,
            'opportunity': opp,
            'rate_per_million': round(rate_per_million, 4),
        })
    
    spectrum_df = pd.DataFrame(spectrum_rows)
    
    total_obs = spectrum_df['observed'].sum()
    if total_obs > 0:
        spectrum_df['percentage'] = round(spectrum_df['observed'] / total_obs * 100, 4)
    else:
        spectrum_df['percentage'] = 0
    
    # Save the three forms
    spectrum_df[['sbs96_channel', 'observed']].to_csv(
        os.path.join(results_dir, 'sbs96_spectrum_raw.csv'), index=False)
    spectrum_df[['sbs96_channel', 'rate_per_million']].to_csv(
        os.path.join(results_dir, 'sbs96_spectrum_per_million.csv'), index=False)
    spectrum_df[['sbs96_channel', 'percentage']].to_csv(
        os.path.join(results_dir, 'sbs96_spectrum_percentage.csv'), index=False)
    
    # Pickle for downstream
    with open(os.path.join(results_dir, 'sbs96_data.pkl'), 'wb') as f:
        pickle.dump({
            'sbs96_counts': dict(sbs96_counts),
            'trinuc_opportunity': dict(trinuc_opp),
            'sbs96_channels': sbs96_channels,
            'total_singletons_assigned': len(sbs96_rows),
        }, f)
    
    cross_summary.append({
        'species': sp,
        'total_singletons_assigned': len(sbs96_rows),
        'channels_with_data': int((spectrum_df['observed'] > 0).sum()),
    })
    
    print('  Saved: sbs96_singletons.csv, sbs96_spectrum_raw.csv,')
    print('         sbs96_spectrum_per_million.csv, sbs96_spectrum_percentage.csv,')
    print('         sbs96_data.pkl')

cross_df = pd.DataFrame(cross_summary)
cross_df.to_csv(os.path.join(base_dir, 'sbs96_summary.csv'), index=False)

print('\n' + '=' * 60)
print('Saved cross-species summary to: ' + os.path.join(base_dir, 'sbs96_summary.csv'))
print(cross_df.to_string(index=False))