'''
Build a small summary CSV with per-species:
  - global dN/dS, Ts/Tv
  - leading-strand dN/dS, Ts/Tv
  - lagging-strand dN/dS, Ts/Tv
'''

import pandas as pd
import pickle
import os

base_dir = r'C:\Users\[USER]\...\Species_folder'

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

# Transitions: A<>G, C<>T
transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

def safe_load_pickle(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def safe_load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

# Compute Ts/Tv from a 4x4 mutation matrix DataFrame
def compute_tstv(matrix):
    if matrix is None:
        return None
    ts = 0
    tv = 0
    for ref in ['A', 'C', 'G', 'T']:
        for alt in ['A', 'C', 'G', 'T']:
            if ref == alt:
                continue
            count = matrix.loc[ref, alt] if ref in matrix.index and alt in matrix.columns else 0
            if (ref, alt) in transitions:
                ts += count
            else:
                tv += count
    if tv == 0:
        return None
    return round(ts / tv, 4)

# Compute strand-stratified dN/dS from codon_singletons.csv + codon_data.pkl
def compute_strand_dnds(codon_singletons_csv, codon_pkl):
    if codon_singletons_csv is None or codon_pkl is None:
        return None, None
    
    syn_opp = codon_pkl.get('syn_opportunity', 0)
    nonsyn_opp = codon_pkl.get('nonsyn_opportunity', 0)
    
    if syn_opp == 0 or nonsyn_opp == 0:
        return None, None
    
    cs = codon_singletons_csv
    if 'effect' not in cs.columns or 'local_strand' not in cs.columns:
        return None, None
    
    leading = cs[cs['local_strand'] == 'leading']
    lagging = cs[cs['local_strand'] == 'lagging']
    
    def calc(sub):
        s = (sub['effect'] == 'synonymous').sum()
        n = (sub['effect'] == 'non-synonymous').sum()
        if s == 0:
            return None
        dn = n / nonsyn_opp
        ds = s / syn_opp
        if ds == 0:
            return None
        return round(dn / ds, 4)
    
    return calc(leading), calc(lagging)

rows = []

for folder, full_name in species_ordered:
    print('Processing: ' + folder)
    
    results_dir = os.path.join(base_dir, folder, 'results')
    
    row = {'Species': full_name}
    
    # Global dN/dS from codon_data.pkl
    codon_pkl = safe_load_pickle(os.path.join(results_dir, 'codon_data.pkl'))
    if codon_pkl is not None:
        dnds = codon_pkl.get('dnds', None)
        row['global_dN_dS'] = round(dnds, 4) if dnds is not None else None
    else:
        row['global_dN_dS'] = None
    
    # Global Ts/Tv from singleton mutation_spectrum_raw.csv
    sing_raw = safe_load_csv(os.path.join(results_dir, 'mutation_spectrum_raw.csv'))
    row['global_Ts_Tv'] = compute_tstv(sing_raw)
    
    # Leading and lagging Ts/Tv
    leading_raw = safe_load_csv(os.path.join(results_dir, 'spectrum_leading_template.csv'))
    lagging_raw = safe_load_csv(os.path.join(results_dir, 'spectrum_lagging_template.csv'))
    row['leading_Ts_Tv'] = compute_tstv(leading_raw)
    row['lagging_Ts_Tv'] = compute_tstv(lagging_raw)
    
    # Leading and lagging dN/dS - need per-singleton codon table
    cs = None
    cs_path = os.path.join(results_dir, 'codon_singletons.csv')
    if os.path.exists(cs_path):
        try:
            cs = pd.read_csv(cs_path)
        except Exception:
            cs = None
    
    lead_dnds, lag_dnds = compute_strand_dnds(cs, codon_pkl)
    row['leading_dN_dS'] = lead_dnds
    row['lagging_dN_dS'] = lag_dnds
    
    rows.append(row)

# Reorder columns the way the user asked
col_order = [
    'Species',
    'global_dN_dS', 'global_Ts_Tv',
    'leading_dN_dS', 'leading_Ts_Tv',
    'lagging_dN_dS', 'lagging_Ts_Tv',
]
out_df = pd.DataFrame(rows)[col_order]

out_path = os.path.join(base_dir, 'dnds_tstv_summary.csv')
out_df.to_csv(out_path, index=False)

print('\nSaved: ' + out_path)
print(out_df.to_string(index=False))
