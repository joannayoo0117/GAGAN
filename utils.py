from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
import numpy as np
import mdtraj as md
import tempfile
import os
from deepchem.feat.mol_graphs import ConvMol

# TODO change atom list to all atoms
_possible_atom_list =  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
    'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt','Hg', 'Pb', 'As', 'UNK']
_possible_numH_list = [0, 1, 2, 3, 4]
_possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
_possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
_possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2]
_possible_number_radical_e_list = [0, 1, 2]
_possible_chirality_list = ['R', 'S']


def one_hot_encoding(x, set):
    if x not in set:
        raise Exception("input {0} not in allowable set{1}:".format(x, set))
    return list(map(lambda s: int(x == s), set))


def one_hot_encoding_unk(x, set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in set:
        x = set[-1]
    return list(map(lambda s: int(x == s), set))


def encode_atom(atom, bool_id_feat=False,
                explicit_H=False, use_chirality=False):
    """
    From deepchem.feat.graph_features
    """

    # why not one-hot get_formal_charge and get_num_radical_electrons?
    result = \
        one_hot_encoding_unk(
            atom.GetSymbol(), _possible_atom_list) + \
        one_hot_encoding(atom.GetDegree(), _possible_degree_list) + \
        one_hot_encoding_unk(
            atom.GetImplicitValence(), _possible_valence_list) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_hot_encoding_unk(
            atom.GetHybridization(), _possible_hybridization_list) + \
        [atom.GetIsAromatic()]

    if not explicit_H:
        result = result + one_hot_encoding_unk(
            atom.GetTotalNumHs(), _possible_numH_list)
    if use_chirality:
        try:
            result = result + one_hot_encoding_unk(
                atom.GetProp('_CIPCode'), _possible_chirality_list) + \
                [atom.HasProp('_Ch_possible_numH_list = iralityPossible')]
        except:
            result = result + [False, False] + \
                [atom.HasProp('_ChiralityPossible')]

    return np.array(result)


def get_molecules_from_pdb(protein_pdb, ligand_pdb):
    """
    Input:
        protein_pdb: str. Path to protein .pdb file
        ligand_pdb: str. Path to ligand .pdb file
    Output:
        protein: rdkit.Chem.rdchem.Mol
        ligand: rdkit.Chem.rdchem.Mol
        compl: rdkit.Chem.rdchem.Mol
    """

    # TODO Currently combining protein and ligand with
    # mdtraj. Verify this method.
    complex_traj = combine_mdtraj(md.load(protein_pdb),
                                  md.load(ligand_pdb))
    tmpfile = tempfile.mkstemp(suffix='.pdb')
    f, comp_pdb = tmpfile

    complex_traj.save(comp_pdb)

    protein = rdmolfiles.MolFromPDBFile(protein_pdb)
    ligand = rdmolfiles.MolFromPDBFile(ligand_pdb)
    compl = rdmolfiles.MolFromPDBFile(comp_pdb)

    os.close(f)
    os.remove(comp_pdb)

    return (protein, ligand, compl)   


def build_graph_from_molecule(mol, use_master_atom=False):
    """
    Param:
        mol - rdkit.Chem.rdchem.Mol
    Output:
        nodes - np.ndarray of shape (num_atoms, num_feat)
        canon_adj_list - list. index corresponds to the index of node
                         and canon_adj_list[index] corresponds to indices
                         of the nodes that node i is connected to.
    """
    if not isinstance(mol, Chem.rdchem.Mol):
        raise TypeError("'mol' must be rdkit.Chem.rdchem.Mol obj")

    # what are the two lines below doing?
    # Answer found in deepchem.data.data_loader featurize_smiles_df
    # TODO (ytz) this is a bandage solution to reorder the atoms so
    # that they're always in the same canonical order. Presumably this
    # should be correctly implemented in the future for graph mols.
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)


    idx_nodes = [(atom.GetIdx(), encode_atom(atom))
                 for atom in mol.GetAtoms()]
    idx_nodes.sort()
    _, nodes = list(zip(*idx_nodes))

    nodes = np.vstack(nodes)

    # Master atom is the "average" of all atoms that is connected to all atom
    # Introduced in https://arxiv.org/pdf/1704.01212.pdf
    if use_master_atom:
        master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)
        nodes = np.concatenate([nodes, master_atom_features], axis=0)

    edge_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in mol.GetBonds()]

    canon_adj_list = [[] for _ in range(len(nodes))]

    for edge in edge_list:
        canon_adj_list[edge[0]].append(edge[1])
        canon_adj_list[edge[1]].append(edge[0])

    if use_master_atom:
        fake_atom_index = len(nodes) - 1

        for i in range(len(nodes) - 1):
            canon_adj_list[i].append(fake_atom_index)

    return (nodes, canon_adj_list)


def combine_mdtraj(protein_traj, ligand_traj):
    """
    TODO Combining molecules this way is not stable.
    Find a better way to do this.
    """
    chain = protein_traj.topology.add_chain()
    residue = protein_traj.topology.add_residue("LIG", chain, resSeq=1)
    for atom in ligand_traj.topology.atoms:
        protein_traj.topology.add_atom(atom.name, atom.element, residue)
    protein_traj.xyz = np.hstack([protein_traj.xyz, ligand_traj.xyz])
    protein_traj.topology.create_standard_bonds()

    return protein_traj


def build_p2l_distance_matrix(protein_adj_list,
                              ligand_adj_list,
                              complex_adj_list,
                              max_distance=5):
    num_protein_atoms = len(protein_adj_list)
    num_ligand_atoms = len(ligand_adj_list)
    num_complex_atoms = num_protein_atoms + num_ligand_atoms

    assert num_complex_atoms == len(complex_adj_list)

    distance_mat = []
    def build_distance_vec(dist_vec, d, max_d,
                           connected_nodes, adjacency_list):
        for node_index in connected_nodes:
            if dist_vec[node_index] == np.inf:
                dist_vec[node_index] = 1
            elif dist_vec[node_index] < np.inf:
                dist_vec[node_index] += 1
        if d == max_d - 1:
            return dist_vec
        else:
            connected_nodes = list(set([node for n in connected_nodes
                                        for node in adjacency_list[n]]))
            return build_distance_vec(
                dist_vec, d+1, max_d, connected_nodes, adjacency_list)

    for i in range(num_ligand_atoms):
        ligand_index = num_protein_atoms + i
        ligand2protein = [j for j in complex_adj_list[ligand_index]
                          if j < num_protein_atoms]

        dist_vec = np.array([np.inf for _ in range(num_protein_atoms)])
        dist_vec_final = build_distance_vec(
            dist_vec, 0, max_distance,
            ligand2protein, protein_adj_list)

        distance_mat.append(dist_vec_final)

    return np.array(distance_mat)


def transform_distance_graph(D, mode='gaussian', mu=0, sigma=10):
    """
    Temporary solution to see if learable parameter for 
    distance-aware adjacency matrix is a good idea
    """
    div = lambda i: 1 / i if i > 0 else 0
    gaussian = lambda i: np.exp(-(i - mu)**2 / sigma) if i > 0 else 0

    f = np.vectorize(eval(mode))
    D = f(D)

    return D


def build_adjacency_matrix(adj_list):
    adjacency_matrix = np.zeros((len(adj_list), len(adj_list)))

    u = []
    v = []
    for node, out_nodes in enumerate(adj_list):
        u.extend([node] * (len(out_nodes) + 1))
        v.append(node)
        v.extend(out_nodes)

    adjacency_matrix[[u, v]] = 1

    return adjacency_matrix


def get_pdb_features(protein_pdb_file,
                     ligand_pdb_file,
                     data_dir='./data/pdbbind/v2018'):
    protein_pdb_file = os.path.join(data_dir, protein_pdb_file)
    ligand_pdb_file = os.path.join(data_dir, ligand_pdb_file)

    if not os.path.exists(protein_pdb_file):
        raise IOError(".pdb file not found in " + protein_pdb_file)
    if not os.path.exists(ligand_pdb_file):
        raise IOError(".pdb file not found in " + ligand_pdb_file)

    
    (protein, ligand, compl) = get_molecules_from_pdb(
        protein_pdb_file, ligand_pdb_file)

    node_feat_p, adj_list_p = build_graph_from_molecule(protein)
    node_feat_l, adj_list_l = build_graph_from_molecule(ligand)
    node_feat_c, adj_list_c = build_graph_from_molecule(compl)

    num_p_atoms = len(adj_list_p)
    num_l_atoms = len(adj_list_l)
    n_feat = node_feat_p.shape[1]

    X_protein = np.concatenate((node_feat_p, np.zeros((num_l_atoms, n_feat))))
    X_ligand = np.concatenate((np.zeros((num_p_atoms, n_feat)), node_feat_l))
    X = np.concatenate((X_protein, X_ligand), axis=1)

    A = build_adjacency_matrix(adj_list_c)

    D = build_p2l_distance_matrix(adj_list_p, adj_list_l, adj_list_c)
    # This is a temporary solution to fix the DAAM
    D = transform_distance_graph(D)
    (dim_l, dim_p) = D.shape

    A2 = np.copy(A)
    A2[-dim_l:, :dim_p] = D
    A2[:dim_p, -dim_l:] = D.transpose() 

    return X, A, A2


def get_convmol_features(protein_pdb_file,
                         ligand_pdb_file,
                         data_dir='./data/pdbbind/v2018'):
    protein_pdb_file = os.path.join(data_dir, protein_pdb_file)
    ligand_pdb_file = os.path.join(data_dir, ligand_pdb_file)

    if not os.path.exists(protein_pdb_file):
        raise IOError(".pdb file not found in " + protein_pdb_file)
    if not os.path.exists(ligand_pdb_file):
        raise IOError(".pdb file not found in " + ligand_pdb_file)

    
    (_, _, compl) = get_molecules_from_pdb(
        protein_pdb_file, ligand_pdb_file)

    # This is from deepchem.models.graph_models.GraphConv 
    # default generator
    nodes, adj_list = build_graph_from_molecule(compl)
    convmol = ConvMol(nodes, adj_list)
    multiConvMol = convmol.agglomerate_mols([convmol])
    (node_feat, deg_slice, membership, deg_adj_list) = \
        (multiConvMol.get_atom_features(), 
         multiConvMol.deg_slice, 
         np.array(multiConvMol.membership),
         multiConvMol.get_deg_adjacency_lists())

    return (node_feat, deg_slice, membership, deg_adj_list)