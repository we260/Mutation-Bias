
import numpy as np
import pandas as pd
import vcfpy
from collections import Counter, defaultdict
from Bio import SeqIO
import pickle
import os


'''
Overview:
This script identifies 'super-singleton'(SNVs) and produces per-species 
mutation spectra normalised against non-recombinant base composition. 

A super-singleton is defined as an SNV occurring in a sample that has 
only one non-recombinant SNV across the entire genome relative to the 
majority consensus.

These represent the rarest and most recently arisen variants in
the dataset and offer the closest approximation to a pre-selective 
view of mutational processes.

Input:
For each of the 18 bacterial species included in this analysis, the
script expects three files:
  - {results}/all_pass.vcf : the PASS-filtered multi-sample VCF
    produced by step 1 of the pipeline
  - {results}/intermediate_data.pkl : the pickled output of the
    upstream pipeline, containing per-sample SNV positions, flagged
    recombinant SNV positions, majority allele assignments, and
    non-recombinant base composition.
  - [raw_data]/[species]_fasta.fna : the reference genome sequence in
    FASTA format.

Method:
For each species:

  (1) Per-sample non-recombinant SNV counting. Using the per-sample
      SNV positions and the recombinant SNV positions from the upstream
      pipeline, each sample's SNV count is recomputed after excluding
      SNVs falling within recombinant territory.

  (2) Super-singleton sample identification. Samples carrying exactly
      one non-recombinant SNV across the whole genome are retained as
      super-singleton samples.

  (3) Base extraction. The all_pass.vcf is read once to extract the
      reference and singleton bases at each super-singleton position,
      and the directional mutation type (e.g. C>T) is recorded.

  (4) Opportunity normalisation. Singleton counts are divided by the
      count of the corresponding reference base in the non-recombinant
      genome composition. Per-million and percentage forms are also
      produced.

'''



base_dir = r'C:\Users\willi\OneDrive\Documents\University\Year 3\Semester 2\Capstone\Designated Species'

species = [
    'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
    'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
    'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
    'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
    'V_cholerae', 'S_typhimurium_2',
]

bases = ['A', 'C', 'G', 'T']

summary_rows = []

for sp in species:
    print('\n' + '=' * 60)
    print('Processing: ' + sp)
    
    species_dir = os.path.join(base_dir, sp)
    raw_data_dir = os.path.join(species_dir, 'raw_data')
    results_dir = os.path.join(species_dir, 'results')
    
    all_pass_path = os.path.join(results_dir, 'all_pass.vcf')
    pickle_path = os.path.join(results_dir, 'intermediate_data.pkl')
    fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fna')
    
    if not os.path.exists(all_pass_path):
        print('  Skipped: no all_pass.vcf')
        continue
    if not os.path.exists(pickle_path):
        print('  Skipped: no pickle')
        continue
    if not os.path.exists(fasta_path):
        print('  Skipped: no FASTA')
        continue
    
    # Load pickled intermediate data
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    majority_alleles = data['majority_alleles']
    snv_pos = data['snv_pos']
    suspicious_snvs_raw = data['suspicious_snvs']
    suspicious_snvs = {s: set(tuple(p) for p in positions) for s, positions in suspicious_snvs_raw.items()}
    nr_base_counts = data.get('nr_base_counts', None)
    
    if nr_base_counts is None:
        print('  SKIPPED - no nr_base_counts in pickle (rerun upstream pipeline)')
        continue
    
    total_samples = len(snv_pos)
    
    # Count non-recombinant SNVs per sample
    sample_snv_counts = {}
    for sample, positions in snv_pos.items():
        recomb_set = suspicious_snvs.get(sample, set())
        non_recomb = [p for p in positions if p not in recomb_set]
        sample_snv_counts[sample] = len(non_recomb)
    
    # Identify super-singleton samples (exactly one non-recombinant SNV)
    super_samples = [s for s, n in sample_snv_counts.items() if n == 1]
    print('  Samples: ' + str(total_samples) + ' | super-singleton samples: ' + str(len(super_samples)))
    
    if len(super_samples) == 0:
        summary_rows.append({
            'species': sp,
            'total_samples': total_samples,
            'super_singleton_samples': 0,
            'super_singletons_found': 0,
        })
        continue
    
    # Build a lookup: position -> list of super-singleton samples
    super_lookup = {}
    for sample in super_samples:
        recomb_set = suspicious_snvs.get(sample, set())
        non_recomb = [p for p in snv_pos[sample] if p not in recomb_set]
        super_lookup[sample] = non_recomb[0]
    
    position_to_sample = {}
    for sample, (chrom, pos) in super_lookup.items():
        position_to_sample.setdefault((chrom, pos), []).append(sample)
    
    # Read VCF to extract REF/ALT bases at super-singleton positions
    reader = vcfpy.Reader.from_path(all_pass_path)
    super_details = []
    
    for record in reader:
        key = (record.CHROM, record.POS)
        if key not in position_to_sample:
            continue
        
        majority_gt = majority_alleles.get(key)
        if majority_gt is None:
            continue
        
        if majority_gt == '0':
            majority_base = record.REF
        else:
            try:
                majority_base = record.ALT[int(majority_gt) - 1].value
            except (IndexError, ValueError):
                continue
        
        target_samples = set(position_to_sample[key])
        for call in record.calls:
            if call.sample not in target_samples:
                continue
            
            gt = call.data.get('GT')
            if gt is None or gt == '.' or gt == majority_gt:
                continue
            
            if gt == '0':
                singleton_base = record.REF
            else:
                try:
                    singleton_base = record.ALT[int(gt) - 1].value
                except (IndexError, ValueError):
                    continue
            
            if len(majority_base) == 1 and len(singleton_base) == 1:
                super_details.append({
                    'chrom': record.CHROM,
                    'pos': record.POS,
                    'sample': call.sample,
                    'majority_base': majority_base,
                    'singleton_base': singleton_base,
                    'mutation_type': majority_base + '>' + singleton_base,
                })
    
    reader.close()
    
    print('  Valid super-singletons: ' + str(len(super_details)))
    
    if len(super_details) == 0:
        summary_rows.append({
            'species': sp,
            'total_samples': total_samples,
            'super_singleton_samples': len(super_samples),
            'super_singletons_found': 0,
        })
        continue
    
    # Save per-singleton details
    pd.DataFrame(super_details).to_csv(
        os.path.join(results_dir, 'super_singleton_details.csv'), index=False)
    
    # Build mutation spectrum matrix
    mutation_counts = Counter(d['mutation_type'] for d in super_details)
    matrix = []
    for ref_base in bases:
        row = []
        for alt_base in bases:
            row.append(mutation_counts.get(ref_base + '>' + alt_base, 0))
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=bases, columns=bases)
    df.index.name = 'FROM (majority)'
    df.columns.name = 'TO (singleton)'
    
    print('\nMutation Spectrum Matrix:')
    print(df)
    print('\nTotal super-singletons: ' + str(df.sum().sum()))
    
    # Standardise: divide each row by non-recombinant ref base count
    df_standardised = df.copy().astype(float)
    for base in bases:
        if nr_base_counts.get(base, 0) > 0:
            df_standardised.loc[base] = df_standardised.loc[base] / nr_base_counts[base]
    
    df_per_million = df_standardised * 1000000
    print('\nMutations per million non-recombinant reference bases:')
    print(df_per_million.round(2))
    
    total_pm = df_per_million.values.sum()
    if total_pm > 0:
        df_pct = (df_per_million / total_pm) * 100
    else:
        df_pct = df_per_million.copy()
    
    print('\nPercentage spectrum:')
    print(df_pct.round(2))
    
    # Save the three spectrum CSVs
    df.to_csv(os.path.join(results_dir, 'super_singleton_spectrum_raw.csv'))
    df_per_million.round(2).to_csv(os.path.join(results_dir, 'super_singleton_spectrum_per_million.csv'))
    df_pct.round(2).to_csv(os.path.join(results_dir, 'super_singleton_spectrum_percentage.csv'))
    
    # Save per-species summary
    with open(os.path.join(results_dir, 'super_singleton_summary.txt'), 'w') as f:
        f.write('Total samples: ' + str(total_samples) + '\n')
        f.write('Super-singleton samples (exactly 1 non-recombinant SNV): '
                + str(len(super_samples)) + '\n')
        f.write('Valid super-singletons recovered: ' + str(len(super_details)) + '\n')
        f.write('\nNon-recombinant base counts: ' + str(nr_base_counts) + '\n')
        f.write('\nMutation type breakdown:\n')
        for mt in sorted(mutation_counts.keys()):
            f.write('  ' + mt + ': ' + str(mutation_counts[mt]) + '\n')
    
    summary_rows.append({
        'species': sp,
        'total_samples': total_samples,
        'super_singleton_samples': len(super_samples),
        'super_singletons_found': len(super_details),
    })
    
    print('\nResults saved to: ' + results_dir)
    print('  - super_singleton_spectrum_raw.csv')
    print('  - super_singleton_spectrum_per_million.csv')
    print('  - super_singleton_spectrum_percentage.csv')
    print('  - super_singleton_summary.txt')
    print('  - super_singleton_details.csv')

# Save cross-species summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(base_dir, 'super_singleton_summary.csv'), index=False)

print('\n' + '=' * 60)
print('Cross-species summary saved to: ' + os.path.join(base_dir, 'super_singleton_summary.csv'))
print(summary_df.to_string(index=False))