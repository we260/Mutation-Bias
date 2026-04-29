'''
========================================================================
Master CSV builder
========================================================================

Compiles all per-species outputs from the upstream pipelines (singleton,
super-singleton, GC skew, codon) into a single wide-format master CSV
with one row per species.

Columns include:
  - genome and recombination metrics
  - singleton totals
  - super-singleton totals
  - dN, dS, dN/dS
  - oriC, terC, leading/lagging singleton counts
  - 12 mutation type counts and percentages (singletons)
  - 12 mutation type counts and percentages (super-singletons)
  - 12 leading-strand and 12 lagging-strand mutation type counts and percentages
  - codon position 1/2/3 totals
  - 64 per-codon singleton counts
  - 64 per-codon genome counts (opportunity)
========================================================================
'''

import pandas as pd
import pickle
import os

base_dir = r'C:\Users\willi\OneDrive\Documents\University\Year 3\Semester 2\Capstone\Designated Species'

# Output order matches user's requested species order
species_ordered = [
    ('A_baumannii', 'Acinetobacter_baumannii'),
    ('B_pertussis', 'Bordetella_pertussis'),
    ('C_jejuni', 'Campylobacter_jejuni'),
    ('C_difficile', 'Clostridioides_difficile'),
    ('E_coli', 'Escherichia_coli'),
    ('H_influenzae', 'Haemophilus_influenzae'),
    ('L_monocytogenes', 'Listeria_monocytogenes'),
    ('M_tuberculosis_2', 'Mycobacterium_tuberculosis'),
    ('N_gonorrhoeae', 'Neisseria_gonorrhoeae'),
    ('N_meningitidis', 'Neisseria_meningitidis'),
    ('P_aeruginosa', 'Pseudomonas_aeruginosa'),
    ('S_aureus', 'Staphylococcus_aureus'),
    ('S_epidermis', 'Staphylococcus_epidermidis'),
    ('S_agalactiae', 'Streptococcus_agalactiae'),
    ('S_pneumoniae', 'Streptococcus_pneumoniae'),
    ('S_pyogenes', 'Streptococcus_pyogenes'),
    ('V_cholerae', 'Vibrio_cholerae'),
    ('S_typhimurium_2', 'Salmonella_typhimurium'),
]

bases = ['A', 'C', 'G', 'T']
mut_types = []
for ref in bases:
    for alt in bases:
        if ref != alt:
            mut_types.append(ref + '>' + alt)

# All 64 codons in alphabetical order
codons = sorted([a + b + c for a in bases for b in bases for c in bases])

def safe_load(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

def safe_load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_load_pickle(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

# Get the matrix value, or 0 if missing
def get_cell(matrix_df, row, col):
    if matrix_df is None:
        return 0
    if row in matrix_df.index and col in matrix_df.columns:
        return float(matrix_df.loc[row, col])
    return 0

# Convert a 4x4 matrix to a 12-element list of (name, value) tuples
def matrix_to_dict(matrix_df, prefix):
    out = {}
    for ref in bases:
        for alt in bases:
            if ref != alt:
                key = prefix + '_' + ref + '_to_' + alt
                out[key] = get_cell(matrix_df, ref, alt)
    return out

rows = []

for folder, full_name in species_ordered:
    print('Processing: ' + folder)
    
    species_dir = os.path.join(base_dir, folder)
    results_dir = os.path.join(species_dir, 'results')
    
    row = {
        'Species': full_name,
        'folder': folder,
    }
    
    # Singleton pickle: genome size, base counts, recombinant territory, singletons
    sing_pkl = safe_load_pickle(os.path.join(results_dir, 'intermediate_data.pkl'))
    if sing_pkl is not None:
        row['genome_size'] = sing_pkl.get('genome_size', None)
        bc = sing_pkl.get('base_counts', {})
        row['genome_A'] = bc.get('A', 0)
        row['genome_C'] = bc.get('C', 0)
        row['genome_G'] = bc.get('G', 0)
        row['genome_T'] = bc.get('T', 0)
        nbc = sing_pkl.get('nr_base_counts', {})
        row['nr_genome_A'] = nbc.get('A', 0)
        row['nr_genome_C'] = nbc.get('C', 0)
        row['nr_genome_G'] = nbc.get('G', 0)
        row['nr_genome_T'] = nbc.get('T', 0)
        row['recombinant_territory_bp'] = sing_pkl.get('recombinant_territory', 0)
        if row['genome_size']:
            row['recombinant_territory_pct'] = round(
                row['recombinant_territory_bp'] / row['genome_size'] * 100, 2)

    # Singleton spectra
    sing_raw = safe_load(os.path.join(results_dir, 'mutation_spectrum_raw.csv'))
    sing_pct = safe_load(os.path.join(results_dir, 'mutation_spectrum_percentage.csv'))
    sing_pm = safe_load(os.path.join(results_dir, 'mutation_spectrum_per_million.csv'))
    
    # If percentage CSV is missing, compute from raw
    if sing_pct is None and sing_raw is not None:
        total = sing_raw.values.sum()
        if total > 0:
            sing_pct = sing_raw / total * 100
        
        
    row['total_singletons'] = int(sing_raw.values.sum()) if sing_raw is not None else 0
    

    row.update(matrix_to_dict(sing_raw, 'singleton_count'))
    row.update(matrix_to_dict(sing_pct, 'singleton_pct'))
    row.update(matrix_to_dict(sing_pm, 'singleton_per_million'))
    
    # Super-singleton spectra
    super_raw = safe_load(os.path.join(results_dir, 'super_singleton_spectrum_raw.csv'))
    super_pct = safe_load(os.path.join(results_dir, 'super_singleton_spectrum_percentage.csv'))
    super_pm = safe_load(os.path.join(results_dir, 'super_singleton_spectrum_per_million.csv'))
    
    row['total_super_singletons'] = int(super_raw.values.sum()) if super_raw is not None else 0
    
    row.update(matrix_to_dict(super_raw, 'super_singleton_count'))
    row.update(matrix_to_dict(super_pct, 'super_singleton_pct'))
    row.update(matrix_to_dict(super_pm, 'super_singleton_per_million'))
    
    # GC skew metadata
    gc_pkl = safe_load_pickle(os.path.join(results_dir, 'gc_skew_data.pkl'))
    if gc_pkl is not None:
        row['oriC'] = gc_pkl.get('oriC', None)
        row['terC'] = gc_pkl.get('terC', None)
        lbc = gc_pkl.get('leading_base_counts', {})
        row['leading_genome_A'] = lbc.get('A', 0)
        row['leading_genome_C'] = lbc.get('C', 0)
        row['leading_genome_G'] = lbc.get('G', 0)
        row['leading_genome_T'] = lbc.get('T', 0)
        labc = gc_pkl.get('lagging_base_counts', {})
        row['lagging_genome_A'] = labc.get('A', 0)
        row['lagging_genome_C'] = labc.get('C', 0)
        row['lagging_genome_G'] = labc.get('G', 0)
        row['lagging_genome_T'] = labc.get('T', 0)
    
    # Leading/lagging spectra
    leading_raw = safe_load(os.path.join(results_dir, 'spectrum_leading_template.csv'))
    leading_pct = safe_load(os.path.join(results_dir, 'spectrum_leading_template_percentage.csv'))
    leading_pm = safe_load(os.path.join(results_dir, 'spectrum_leading_template_per_million.csv'))
    lagging_raw = safe_load(os.path.join(results_dir, 'spectrum_lagging_template.csv'))
    lagging_pct = safe_load(os.path.join(results_dir, 'spectrum_lagging_template_percentage.csv'))
    lagging_pm = safe_load(os.path.join(results_dir, 'spectrum_lagging_template_per_million.csv'))
    
    row['n_leading_singletons'] = int(leading_raw.values.sum()) if leading_raw is not None else 0
    row['n_lagging_singletons'] = int(lagging_raw.values.sum()) if lagging_raw is not None else 0
    
    row.update(matrix_to_dict(leading_raw, 'leading_count'))
    row.update(matrix_to_dict(leading_pct, 'leading_pct'))
    row.update(matrix_to_dict(leading_pm, 'leading_per_million'))
    row.update(matrix_to_dict(lagging_raw, 'lagging_count'))
    row.update(matrix_to_dict(lagging_pct, 'lagging_pct'))
    row.update(matrix_to_dict(lagging_pm, 'lagging_per_million'))
    
    # Codon data
    codon_pkl = safe_load_pickle(os.path.join(results_dir, 'codon_data.pkl'))
    if codon_pkl is not None:
        row['cds_features'] = len(codon_pkl.get('cds_features', []))
        cbc = codon_pkl.get('cds_base_counts', {})
        row['cds_A'] = cbc.get('A', 0)
        row['cds_C'] = cbc.get('C', 0)
        row['cds_G'] = cbc.get('G', 0)
        row['cds_T'] = cbc.get('T', 0)
        ibc = codon_pkl.get('intergenic_base_counts', {})
        row['intergenic_A'] = ibc.get('A', 0)
        row['intergenic_C'] = ibc.get('C', 0)
        row['intergenic_G'] = ibc.get('G', 0)
        row['intergenic_T'] = ibc.get('T', 0)
        row['syn_opportunity'] = codon_pkl.get('syn_opportunity', 0)
        row['nonsyn_opportunity'] = codon_pkl.get('nonsyn_opportunity', 0)
        row['stop_opportunity'] = codon_pkl.get('stop_opportunity', 0)
        row['syn_count'] = codon_pkl.get('syn_count', 0)
        row['nonsyn_count'] = codon_pkl.get('nonsyn_count', 0)
        row['stop_count'] = codon_pkl.get('stop_count', 0)
        row['dN'] = codon_pkl.get('dn', None)
        row['dS'] = codon_pkl.get('ds', None)
        row['dN_dS'] = codon_pkl.get('dnds', None)
        
        # Per-codon genome counts (opportunity)
        codon_genome = codon_pkl.get('codon_count_genome', {})
        for c in codons:
            row['codon_genome_count_' + c] = codon_genome.get(c, 0)
    
    # Codon position summary
    cps = safe_load_csv(os.path.join(results_dir, 'codon_position_summary.csv'))
    if cps is not None:
        for cp in [1, 2, 3]:
            sub = cps[cps['codon_position'] == cp]
            if len(sub) > 0:
                row['codon_pos' + str(cp) + '_total'] = int(sub['singleton_count_total'].iloc[0])
                row['codon_pos' + str(cp) + '_leading'] = int(sub['singleton_count_leading'].iloc[0])
                row['codon_pos' + str(cp) + '_lagging'] = int(sub['singleton_count_lagging'].iloc[0])
            else:
                row['codon_pos' + str(cp) + '_total'] = 0
                row['codon_pos' + str(cp) + '_leading'] = 0
                row['codon_pos' + str(cp) + '_lagging'] = 0
    
    # Per-codon singleton counts
    cs = safe_load_csv(os.path.join(results_dir, 'codon_summary.csv'))
    if cs is not None:
        cs_lookup = dict(zip(cs['codon'], cs['singleton_count_total']))
        cs_lead = dict(zip(cs['codon'], cs['singleton_count_leading']))
        cs_lag = dict(zip(cs['codon'], cs['singleton_count_lagging']))
        for c in codons:
            row['codon_singleton_total_' + c] = int(cs_lookup.get(c, 0))
            row['codon_singleton_leading_' + c] = int(cs_lead.get(c, 0))
            row['codon_singleton_lagging_' + c] = int(cs_lag.get(c, 0))
    
    rows.append(row)

# Write master CSV
master_df = pd.DataFrame(rows)
out_path = os.path.join(base_dir, 'master_species_data.csv')
master_df.to_csv(out_path, index=False)

print('\n' + '=' * 60)
print('Saved master CSV: ' + out_path)
print('Rows: ' + str(len(master_df)))
print('Columns: ' + str(len(master_df.columns)))