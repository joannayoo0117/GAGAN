import os
import re
import numpy as np
import pandas as pd 
import requests
import lxml.html as lh
from urllib.request import urlretrieve

"""
These functions were used to generate csv files.
This script is for record-keeping only.
"""
_download_optional_data=False

data_dir = '.'
_all_pdb_ids = os.listdir(os.path.join(data_dir, 'pdbbind/v2018'))

with open(os.path.join(data_dir, './pdbbind/v2018/index/INDEX_refined_data.2018')) as f:
    refinedid2kdki = {line.split()[0]: float(line.split()[3]) for line in f.readlines() 
                      if line[0] != "#"}
unit2scale = {'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9, 'pM': 1e-9, 'fM': 1e-12}

source2type = {'INDEX_general_PL.2018': 'protein-ligand', 
                  'INDEX_general_PN.2018': 'protein-nucleic acid', 
                  'INDEX_general_NL.2018': 'nucleic acid-ligand',
                  'INDEX_general_PP.2018': 'protein-protein'}

def is_protein_id(id, ids=_all_pdb_ids):
    return id in ids


def is_refined_data(id, refined_data = refinedid2kdki):
    return id in refined_data


def process_pdb_index_file(index_file):
    id2affinity = []
    
    with open(index_file, 'r') as f:
        lines = f.readlines()
  
    for line in lines:
        if not is_protein_id(line[:4]):
            continue
            
        line = line.split('//')[0]

        id, resolution, release_year, binding_data = line.split()
        try: 
            resolution = float(resolution)
        except: 
            resolution = None
        try:
            release_year = int(release_year)
        except:
            release_year = None
        interaction_type = source2type[index_file.split('/')[-1]]
        kdki = refinedid2kdki.get(id)


        id2affinity.append(dict(zip(
            ['id', 'resolution', 'release_year', 'binding_data', 'interaction_type', '-log(kd/ki)'],
            [id, resolution, release_year, binding_data, interaction_type, kdki])))
        
    return id2affinity


def get_binding_type(binding_data):
    # Ki=IC50/(1+([L]/Kd)
    if binding_data[:2].lower() == 'kd': return 'kd'
    elif binding_data[:2].lower() == 'ki': return 'ki'
    else: return 'ic50'
    

def get_neg_log_binding_affinity(binding_data):
    binding_affinity_text = re.split('[\=\>\<\~]', binding_data)[-1]
    num, unit = binding_affinity_text[:-2], binding_affinity_text[-2:]
    
    binding_affinity = float(num) * unit2scale.get(unit, 0)

    return -np.log(binding_affinity) if binding_affinity > 0 else 0


def parse_dude_table_from_url(url, out_csv):
    response = requests.get(url)

    table = response.content.split(b'\n')
    start_pos = [i for i, row in enumerate(table) 
                 if '<table>' in str(row)][0]
    end_pos = [i for i, row in enumerate(table) 
               if '</table>' in str(row) and i > start_pos][0]

    doc = lh.fromstring(b'\n'.join(table[start_pos:end_pos+1]))
    tr_elements = doc.xpath('//tr')

    header, content = tr_elements[0], tr_elements[1:]

    columns = [h.strip() for h in header.text_content().split('\n')]

    csv  = []
    for row in content:
        items = [item.strip() for item in row.text_content().split('\n')]
        csv.append(dict(zip(columns, items)))

    dude = pd.DataFrame(csv)

    lower = lambda x: x.lower()
    dude['Target Name'] = dude['Target Name'].apply(lower)

    dude.to_csv(out_csv)

    return dude


def process_pdbbind_dataframe(id2affinity, out_csv):
    binding_affinity_df = pd.DataFrame(id2affinity)

    binding_affinity_df['binding_type'] = \
        binding_affinity_df['binding_data'].apply(get_binding_type)
    binding_affinity_df['binding_affinity'] = \
        binding_affinity_df['binding_data'].apply(get_neg_log_binding_affinity)
    binding_affinity_df = binding_affinity_df.drop('binding_data', axis=1)

    binding_affinity_df.to_csv(out_csv)

    return binding_affinity_df


def save_dude2pdb(out_csv,
                  dude_csv='./data/dude_info.csv', 
                  pdbbind_csv='./data/pdbbind_binding_affinity.csv'):
    dude = pd.read_csv(dude_csv)
    pdbbind = pd.read_csv(pdbbind_csv)

    dude = dude.rename(columns={'Target Name': 'dude_id', 'PDB': 'id'})
    joined = pd.merge(pdbbind, dude[['dude_id', 'id']], on='id')

    return joined[['dude_id', 'id']].to_csv(out_csv)

def main():
    # Download DeepChem data
    deepchem_dataset = ['tox21.csv.gz']
    if _download_optional_data:
        deepchem_dataset.extend(['full_smiles_labels.csv', 
        'core_smiles_labels.csv', 'refined_smiles_labels.csv'])

    deepchem_data_url = \
            'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets'

    for fname in deepchem_dataset:
        urlretrieve(os.path.join(deepchem_data_url, fname),
                    os.path.join(data_dir, fname))
        
    for gz in os.listdir(data_dir):
        if gz.endswith('.gz'):
            os.system('gunzip {}'.format(os.path.join(data_dir, gz)))

    # Process PDBBind database
    id2affinity = process_pdb_index_file(
        os.path.join(data_dir, 'pdbbind/v2018/index/INDEX_general_PL.2018'))
    process_pdbbind_dataframe(id2affinity, 'pdbbind_binding_affinity.csv')

    # Parse DUD-E table 
    parse_dude_table_from_url('http://dude.docking.org/targets', 'dude_info.csv')
    save_dude2pdb('dude2pdb.csv')

if __name__ == "__main__":
    main()