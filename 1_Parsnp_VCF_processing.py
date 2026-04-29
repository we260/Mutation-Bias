import numpy as np
import pandas as pd
import vcfpy
from collections import Counter, defaultdict
from Bio import SeqIO
import pickle
import os

'''
Overview
--------
This script processes Parsnp variant call files (VCFs) from whole-genome
alignments of bacterial isolates and produces per-species mutation
spectra, normalised against non-recombinant base composition. The
pipeline is designed to identify pre-selective mutational biases by
isolating low frequency variants (singletons) and excluding regions
likely to have been introduced via homologous recombination.

For each species:

  (1) PASS filtering. Records flagged as PASS in the parsnp VCF are
      retained; all other records are discarded. The filtered VCF is
      written to {results}/all_pass.vcf for use in downstream
      analyses.

  (2) Majority allele determination. At each variable position, the
      most frequent non-missing genotype across all samples is taken
      as the majority allele. This serves as the consensus reference
      against which sample-specific variants are defined.

  (3) Single nucleotide variant (SNV) identification. For each sample,
      any genotype call that differs from the majority allele at a
      given position is recorded as a candidate SNV.

  (4) Recombination filtering. Recombinant SNVs are identified using
      a sliding-window density approach. A window size of 2,000 bp,
      step size of 1 bp, and a threshold of >=4 SNVs per window were
      used; SNVs falling within any window meeting this threshold
      were classified as recombinant and excluded from further
      analysis. 

  (5) Singleton extraction. After recombination filtering, singletons
      are defined as non-majority genotype calls present in exactly
      one sample at a given position. The mutation type of each
      singleton is recorded as the directional change from the
      majority base to the singleton base (e.g. C>T).

  (6) Opportunity normalisation. The reference genome base composition
      is recomputed using only positions outside recombinant
      territory. Singleton counts are then divided by the count of
      the corresponding reference base in this non-recombinant
      composition, yielding mutation rates per non-recombinant
      reference base. Per-million and percentage forms are also
      produced.'''



# Path to directory containing species data
# base_dir directory should point to the folder containing all folders for each species
# species folders should contain 2 folders: 'raw_data' and 'results'
# raw_data requires 3 items, _parsnp.vcf,  _fasta.fna ,and _genomic.gff

base_dir = r'C:\Users\[USER]\...\Species_folder'
"C:\Users\willi\OneDrive\Documents\University\Year 3\Semester 2\Capstone\Designated Species"

# The name of each species folder within the base_dir path
species = [
    'A_baumannii', 'B_pertussis', 'C_jejuni', 'C_difficile', 'E_coli',
    'H_influenzae', 'L_monocytogenes', 'M_tuberculosis_2',
    'N_gonorrhoeae', 'N_meningitidis', 'P_aeruginosa', 'S_aureus',
    'S_epidermis', 'S_agalactiae', 'S_pneumoniae', 'S_pyogenes',
    'V_cholerae', 'S_typhimurium_2',
]

bases = ['A', 'C', 'G', 'T']
threshold = 4
window_length = 2000

for sp in species:
    print('\n' + '=' * 60)
    print('Processing: ' + sp)
    
    species_dir = os.path.join(base_dir, sp)
    raw_data_dir = os.path.join(species_dir, 'raw_data')
    results_dir = os.path.join(species_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    parsnp_path = os.path.join(raw_data_dir, sp + '_parsnp.vcf')
    fasta_path = os.path.join(raw_data_dir, sp + '_fasta.fna')
    all_pass_path = os.path.join(results_dir, 'all_pass.vcf')
    
    if not os.path.exists(parsnp_path):
        print('  SKIPPED - no parsnp VCF')
        continue
    if not os.path.exists(fasta_path):
        print('  SKIPPED - no FASTA')
        continue
    
    # Calculate genome size and base counts from FASTA
    genome_size = 0
    base_counts = Counter()
    full_seq_by_contig = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        s = str(record.seq).upper()
        full_seq_by_contig[record.id] = s
        genome_size += len(s)
        base_counts += Counter(s)
    base_counts = dict(base_counts)
    print('Genome size: ' + str(genome_size) + ' bp')
    print('Base counts: ' + str(base_counts))
    
    # Filter parsnp VCF for PASS records only -> all_pass.vcf
    parsnp_reader = vcfpy.Reader.from_path(parsnp_path)
    writer = vcfpy.Writer.from_path(all_pass_path, parsnp_reader.header)
    
    passed = 0
    failed = 0
    
    for record in parsnp_reader:
        if 'PASS' in record.FILTER:
            record.INFO = {}
            writer.write_record(record)
            passed += 1
        else:
            failed += 1
    
    parsnp_reader.close()
    writer.close()
    
    print('Records passed: ' + str(passed))
    print('Records failed: ' + str(failed))
    
    # Determine majority allele at each position
    all_pass_reader = vcfpy.Reader.from_path(all_pass_path)
    
    majority_alleles = {}
    majority_tally = defaultdict(int)
    total_positions = 0
    
    for record in all_pass_reader:
        total_positions += 1
        genotypes = []
        for call in record.calls:
            gt = call.data.get('GT')
            if gt and gt != '.':
                genotypes.append(gt)
        
        if genotypes:
            majority_gt = Counter(genotypes).most_common(1)[0][0]
            majority_alleles[(record.CHROM, record.POS)] = majority_gt
            majority_tally[majority_gt] += 1
    
    all_pass_reader.close()
    
    print('total positions in all_pass VCF: ' + str(total_positions))
    print('Majority allele breakdown:')
    for gt, count in sorted(majority_tally.items()):
        print('--> GT type: ' + str(gt) + ': ' + str(count) + ' positions')
    
    # Identify SNVs per sample (any call that deviates from the majority)
    all_pass_reader = vcfpy.Reader.from_path(all_pass_path)
    
    snv_pos = {}
    
    for record in all_pass_reader:
        majority_gt = majority_alleles.get((record.CHROM, record.POS))
        
        for call in record.calls:
            gt = call.data.get('GT')
            
            if gt == '.':
                continue
            
            if gt != majority_gt:
                sample = call.sample
                snv_pos.setdefault(sample, []).append((record.CHROM, record.POS))
    
    all_pass_reader.close()
    
    total_snvs = sum(len(positions) for positions in snv_pos.values())
    print('Total SNV calls before recombination removal: ' + str(total_snvs))
    
    # Singletons before recombinant SNV removal
    all_pass_reader = vcfpy.Reader.from_path(all_pass_path)
    
    pre_removal_singletons = 0
    
    for record in all_pass_reader:
        call_counts = {}
        
        for call in record.calls:
            gt = call.data.get('GT')
            if gt == '.':
                continue
            call_counts[gt] = call_counts.get(gt, 0) + 1
        
        singleton_calls = {gt for gt, count in call_counts.items() if count == 1}
        
        if singleton_calls:
            majority_gt = max(call_counts, key=call_counts.get)
            
            for s_gt in singleton_calls:
                if s_gt != majority_gt:
                    pre_removal_singletons += 1
    
    all_pass_reader.close()
    
    print('Singleton SNVs before recombination removal: ' + str(pre_removal_singletons))
    
    # Identify recombinant SNVs using sliding-window density approach
    suspicious_snvs = {}
    
    for sample, positions in snv_pos.items():
        by_chrom = {}
        for chrom, pos in positions:
            by_chrom.setdefault(chrom, []).append(pos)
        
        suspicious = set()
        
        for chrom, chrom_positions in by_chrom.items():
            chrom_positions = sorted(chrom_positions)
            
            for start_pos in chrom_positions:
                snps_in_window = [p for p in chrom_positions
                                 if start_pos <= p < (start_pos + window_length)]
                
                if len(snps_in_window) >= threshold:
                    for p in snps_in_window:
                        suspicious.add((chrom, p))
        
        suspicious_snvs[sample] = suspicious
    
    total_recombinant = sum(len(positions) for positions in suspicious_snvs.values())
    
    print('Sliding window length: ' + str(window_length))
    print('Threshold (greater than or equal to): ' + str(threshold))
    print('Total recombinant SNV calls flagged: ' + str(total_recombinant))
    
    # Calculate genomic territory covered by recombinant regions
    all_recombinant_spans = set()
    
    for sample, positions in suspicious_snvs.items():
        by_chrom = {}
        for chrom, pos in positions:
            by_chrom.setdefault(chrom, []).append(pos)
        
        for chrom, chrom_positions in by_chrom.items():
            chrom_positions = sorted(chrom_positions)
            
            region_start = chrom_positions[0]
            region_end = chrom_positions[0]
            
            for p in chrom_positions[1:]:
                if p - region_end <= window_length:
                    region_end = p
                else:
                    for bp in range(region_start, region_end + 1):
                        all_recombinant_spans.add((chrom, bp))
                    region_start = p
                    region_end = p
            
            for bp in range(region_start, region_end + 1):
                all_recombinant_spans.add((chrom, bp))
    
    recombinant_territory = len(all_recombinant_spans)
    
    print('Recombinant territory: ' + str(recombinant_territory) + ' bp')
    print('Percentage of genome in recombinant regions: '
          + str(round(recombinant_territory / genome_size * 100, 2)) + '%')
    
    # Calculate non-recombinant base composition
    nr_base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for chrom, seq in full_seq_by_contig.items():
        for i in range(len(seq)):
            pos_1based = i + 1
            if (chrom, pos_1based) in all_recombinant_spans:
                continue
            base = seq[i]
            if base in nr_base_counts:
                nr_base_counts[base] += 1
    
    print('Non-recombinant base counts: ' + str(nr_base_counts))
    
    # Read non-recombinant data, identify singletons, build mutation spectrum
    all_pass_reader = vcfpy.Reader.from_path(all_pass_path)
    
    mutation_counts = defaultdict(int)
    total_singletons = 0
    reversion_count = 0
    singleton_gt_tally = defaultdict(int)
    singleton_details = []
    total_snvs_no_recombination = 0
    
    for record in all_pass_reader:
        # Skip recombinant positions for any sample - mask their GTs to '.'
        call_counts = {}
        sample_by_gt = defaultdict(list)
        
        for call in record.calls:
            sample = call.sample
            gt = call.data.get('GT')
            
            # Skip if missing
            if gt is None or gt == '.':
                continue
            
            # Skip if this sample's call at this position is recombinant
            if sample in suspicious_snvs:
                if (record.CHROM, record.POS) in suspicious_snvs[sample]:
                    continue
            
            call_counts[gt] = call_counts.get(gt, 0) + 1
            sample_by_gt[gt].append(sample)
        
        if not call_counts:
            continue
        
        majority_gt = majority_alleles.get((record.CHROM, record.POS))
        if majority_gt is None:
            continue
        
        for gt, count in call_counts.items():
            if gt != majority_gt:
                total_snvs_no_recombination += count
        
        singleton_calls = {gt for gt, count in call_counts.items() if count == 1}
        
        if singleton_calls:
            if majority_gt == '0':
                majority_base = record.REF
            else:
                majority_base = record.ALT[int(majority_gt) - 1].value
                reversion_count += 1
            
            for s_gt in singleton_calls:
                if s_gt == majority_gt:
                    continue
                
                singleton_gt_tally[s_gt] += 1
                
                if s_gt == '0':
                    singleton_base = record.REF
                else:
                    singleton_base = record.ALT[int(s_gt) - 1].value
                
                if len(majority_base) == 1 and len(singleton_base) == 1:
                    mutation_type = majority_base + '>' + singleton_base
                    mutation_counts[mutation_type] += 1
                    total_singletons += 1
                    
                    singleton_details.append({
                        'chrom': record.CHROM,
                        'pos': record.POS,
                        'sample': sample_by_gt[s_gt][0],
                        'majority_gt': majority_gt,
                        'singleton_gt': s_gt,
                        'majority_base': majority_base,
                        'singleton_base': singleton_base,
                        'mutation_type': mutation_type,
                    })
    
    all_pass_reader.close()
    
    print('Total SNV calls after recombination removal: ' + str(total_snvs_no_recombination))
    print('Reversions: ' + str(reversion_count))
    print('Singleton GT breakdown:')
    for gt, count in sorted(singleton_gt_tally.items()):
        print("  GT '" + str(gt) + "': " + str(count))
    print('Total singleton SNVs: ' + str(total_singletons))
    
    # Build the mutation spectrum matrix
    matrix = []
    for ref_base in bases:
        row = []
        for alt_base in bases:
            mutation = ref_base + '>' + alt_base
            row.append(mutation_counts[mutation])
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=bases, columns=bases)
    df.index.name = 'FROM (majority)'
    df.columns.name = 'TO (singleton)'
    
    print('\nMutation Spectrum Matrix:')
    print(df)
    print('\nTotal mutations: ' + str(df.sum().sum()))
    
    # Standardise: divide each row by the count of that ref base in non-recombinant region
    df_standardised = df.copy().astype(float)
    for base in bases:
        if nr_base_counts[base] > 0:
            df_standardised.loc[base] = df_standardised.loc[base] / nr_base_counts[base]
    
    df_per_million = df_standardised * 1000000
    print('\nMutations per million non-recombinant reference bases:')
    print(df_per_million.round(2))
    
    # Percentage version
    total_pm = df_per_million.values.sum()
    if total_pm > 0:
        df_pct = (df_per_million / total_pm) * 100
    else:
        df_pct = df_per_million.copy()
    
    print('\nPercentage spectrum:')
    print(df_pct.round(2))
    
    # Save the three spectrum CSVs
    df.to_csv(os.path.join(results_dir, 'mutation_spectrum_raw.csv'))
    df_per_million.round(2).to_csv(os.path.join(results_dir, 'mutation_spectrum_per_million.csv'))
    df_pct.round(2).to_csv(os.path.join(results_dir, 'mutation_spectrum_percentage.csv'))
    
    # Save summary stats
    with open(os.path.join(results_dir, 'summary_stats.txt'), 'w') as f:
        f.write('Records passed FILTER: ' + str(passed) + '\n')
        f.write('Records failed FILTER: ' + str(failed) + '\n')
        f.write('Total positions in all_pass VCF: ' + str(total_positions) + '\n')
        f.write('Majority allele breakdown:\n')
        for gt, count in sorted(majority_tally.items()):
            f.write('  GT type ' + str(gt) + ': ' + str(count) + ' positions\n')
        f.write('\nTotal SNV calls before recombination removal: ' + str(total_snvs) + '\n')
        f.write('Singleton SNVs before recombination removal: ' + str(pre_removal_singletons) + '\n')
        f.write('\nSliding window length: ' + str(window_length) + '\n')
        f.write('Threshold (>= SNVs in window): ' + str(threshold) + '\n')
        f.write('Total recombinant SNV calls flagged: ' + str(total_recombinant) + '\n')
        f.write('\nGenome size: ' + str(genome_size) + ' bp\n')
        f.write('Whole-genome base counts: ' + str(base_counts) + '\n')
        f.write('Non-recombinant base counts: ' + str(nr_base_counts) + '\n')
        f.write('Recombinant territory: ' + str(recombinant_territory) + ' bp\n')
        f.write('Percentage of genome in recombinant regions: '
                + str(round(recombinant_territory / genome_size * 100, 2)) + '%\n')
        f.write('\nTotal SNV calls after recombination removal: '
                + str(total_snvs_no_recombination) + '\n')
        f.write('Reversions: ' + str(reversion_count) + '\n')
        f.write('Total singleton SNVs: ' + str(total_singletons) + '\n')
        f.write('\nSingleton GT breakdown:\n')
        for gt, count in sorted(singleton_gt_tally.items()):
            f.write("  GT '" + str(gt) + "': " + str(count) + '\n')
    
    # Save the pickle
    with open(os.path.join(results_dir, 'intermediate_data.pkl'), 'wb') as f:
        pickle.dump({
            'majority_alleles': majority_alleles,
            'majority_tally': dict(majority_tally),
            'snv_pos': snv_pos,
            'suspicious_snvs': {s: list(v) for s, v in suspicious_snvs.items()},
            'mutation_counts': dict(mutation_counts),
            'singleton_gt_tally': dict(singleton_gt_tally),
            'singleton_details': singleton_details,
            'genome_size': genome_size,
            'base_counts': base_counts,
            'nr_base_counts': nr_base_counts,
            'recombinant_territory': recombinant_territory,
        }, f)
    
    print('\nResults saved to: ' + results_dir)
    print('  - mutation_spectrum_raw.csv')
    print('  - mutation_spectrum_per_million.csv')
    print('  - mutation_spectrum_percentage.csv')
    print('  - summary_stats.txt')
    print('  - intermediate_data.pkl')