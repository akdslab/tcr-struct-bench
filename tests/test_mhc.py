"""IMGT numbering of the MHC chains, checked against the helix data we already had.

The numbering itself is tcrkit's (tcrkit.gdomain, tested there); these tests check it
on structures and against the two helix definitions.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from pdbframe import PDBFrame
from tcrkit.gdomain import helix_genes, number_gdomains

import tcrstruct as tcs
from tcrstruct import TCRFrame
from tcrstruct import mhc

DATA = Path(__file__).parent / 'data'

STRUCTURES = ['1AO7/rcsb.pdb', '6ZKW/rcsb.pdb', '7N2N/rcsb.pdb', '3QIB/rcsb.pdb',
              '7T2D/rcsb.pdb', '1AO7/tcrmodel2.pdb', '3QIB/af3-tbm.cif', '7T2D/boltz-2.pdb']

#: References numbered as HLA-A2 / HLA-DR, so their file numbers are the Pierce lab's.
STANDARD_NUMBERED = ['1AO7/rcsb.pdb', '6ZKW/rcsb.pdb', '7N2N/rcsb.pdb', '3QIB/rcsb.pdb',
                     '7T2D/rcsb.pdb']


@pytest.fixture(scope='module')
def structures():
    return {p: TCRFrame.read(DATA / p) for p in STRUCTURES}


def _number(label: str) -> int:
    return int(re.match(r'-?\d+', label).group())


# ---------------------------------- agreement with the Motif Atlas helix ranges

@pytest.mark.parametrize('gene', list(helix_genes()['gene']))
def test_motif_atlas_helices_are_imgt_helices(gene):
    """Every Motif Atlas helix range (the 'v1' helices) lies on the IMGT
    helix: at most its first or last residue is outside (DQA1 starts at 49, the
    end of strand D)."""
    row = helix_genes().loc[gene]
    num = number_gdomains(row['sequence'])
    for start, end in row['helix_ind']:                    # 1-based, inclusive
        part = num.iloc[start - 1:end]
        outside = np.flatnonzero((part['region'] != 'helix').to_numpy())
        assert set(outside) <= {0, len(part) - 1}, part[part['region'] != 'helix']
        on_helix = part[part['region'] == 'helix']
        assert on_helix['domain'].nunique() == 1 and len(on_helix) >= len(part) - 1


def test_motif_atlas_class_ii_helices_shift_by_gene():
    """Why 'v1' and 'imgt' differ on class II: in IMGT numbering the Motif
    Atlas alpha helix starts at a different position for each gene."""
    starts = {}
    for gene in ['DRA', 'DPA1', 'DQA1', 'AA', 'EA']:
        row = helix_genes().loc[gene]
        num = number_gdomains(row['sequence'])
        starts[gene] = num['imgt'].iloc[row['helix_ind'][0][0] - 1]
    assert starts == {'DRA': '52', 'DPA1': '50', 'DQA1': '49', 'AA': '51', 'EA': '52'}


# ------------------------------------------------------------ Pierce helices

@pytest.mark.parametrize('path', STANDARD_NUMBERED)
def test_pierce_helices_are_the_pierce_residue_numbers(structures, path):
    """Found by IMGT position, the Pierce helices are exactly the residue numbers the
    Pierce lab's tool uses (PIERCE_HELICES) in files with standard numbering."""
    pf = structures[path]
    res = pf.df[mhc.mhc_helices(pf)].drop_duplicates(['chain', 'observed_index'])
    ranges = mhc.PIERCE_HELICES[pf.mhc_class]
    want = {(c, n) for c, rs in ranges.items() for lo, hi in rs for n in range(lo, hi + 1)}
    assert set(zip(res['chain'], res['residue_number'])) == want


@pytest.mark.parametrize('path', STRUCTURES)
def test_pierce_helices_follow_the_imgt_numbering(structures, path):
    pf = structures[path]
    num = mhc.mhc_numbering(pf)
    want = set(zip(*num.loc[num['helix'].astype(bool), ['chain', 'observed_index']]
                   .to_numpy().T))
    df = pf.df[mhc.mhc_helices(pf, method='imgt')]
    assert set(zip(df['chain'], df['observed_index'])) == want
    assert set(num.loc[num['helix'].astype(bool), 'region']) == {'helix'}


#: The groove-helix residues by method, as file residue numbers of the references:
#: ``{pdb: {method: {chain: [(first, last), ...]}}}``. On class I both are the Pierce
#: residues; on class II the Motif Atlas ('v1') ranges are shifted by gene.
HELIX_RANGES = {
    '1AO7': {'imgt': {'A': [(50, 86), (140, 176)]}, 'v1': {'A': [(50, 86), (140, 176)]}},
    '6ZKW': {'imgt': {'A': [(50, 86), (140, 176)]}, 'v1': {'A': [(50, 86), (140, 176)]}},
    '7N2N': {'imgt': {'A': [(50, 86), (140, 176)]}, 'v1': {'A': [(50, 86), (140, 176)]}},
    '3QIB': {'imgt': {'A': [(46, 78)], 'B': [(54, 64), (67, 91)]},
             'v1': {'A': [(48, 80)], 'B': [(58, 68), (71, 95)]}},
    '7T2D': {'imgt': {'A': [(46, 78)], 'B': [(54, 64), (67, 91)]},
             'v1': {'A': [(46, 78)], 'B': [(58, 68), (71, 95)]}},
}


def _helix_residues(pf, method):
    ca = pf.df[(pf.df['atom_name'] == 'CA').to_numpy()]
    sel = ca[mhc.mhc_helices(pf, method=method)[ca.index.to_numpy()]]
    return set(zip(sel['chain'], sel['residue_number']))


def _expand(ranges):
    return {(c, n) for c, rs in ranges.items() for lo, hi in rs for n in range(lo, hi + 1)}


@pytest.mark.parametrize('pdb', HELIX_RANGES)
@pytest.mark.parametrize('method', mhc.HELIX_METHODS)
def test_helix_residues_by_method(structures, pdb, method):
    assert _helix_residues(structures[f'{pdb}/rcsb.pdb'], method) == \
        _expand(HELIX_RANGES[pdb][method])


#: The same helices as sequences, one string per helix: class I (HLA-A2, HLA-B), mouse
#: class II (I-E k, 3QIB) and human class II (7T2D).
HELIX_SEQUENCES = {
    '1AO7': {'imgt': {'A': ['PWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYN',
                            'AQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGK']}},
    '6ZKW': {'imgt': {'A': ['PWMEQEGSEYWDRETRSARDTAQIFRVNLRTLRGYYN',
                            'AQISEQKSNDASEAEHQRAYLEDTCVEWLHKYLEKGK']}},
    '3QIB': {'imgt': {'A': ['EEFAKFASFEAQGALANIAVDKANLDVMKERSN'],
                      'B': ['GRPDAENWNSQ', 'FLEQKRAEVDTVCRHNYEIFDNFLV']},
             'v1': {'A': ['FAKFASFEAQGALANIAVDKANLDVMKERSNNT'],
                           'B': ['AENWNSQPEFL', 'KRAEVDTVCRHNYEIFDNFLVPRRV']}},
    '7T2D': {'imgt': {'A': ['EEFGQAFSFEAQGGLANIAILNNNLNTLIQRSN'],
                      'B': ['GRPAAEYWNSQ', 'ILEEKRAVPDRMCRHNYELGGPMTL']},
             'v1': {'A': ['EEFGQAFSFEAQGGLANIAILNNNLNTLIQRSN'],
                           'B': ['AEYWNSQKDIL', 'KRAVPDRMCRHNYELGGPMTLQRRV']}},
}
for _methods in HELIX_SEQUENCES.values():
    _methods.setdefault('v1', _methods['imgt'])   # class I: the methods agree


def _helix_sequences(pf, method):
    ca = pf.df[(pf.df['atom_name'] == 'CA').to_numpy() & mhc.mhc_helices(pf, method=method)]
    out = {}
    for chain, g in ca.groupby('chain'):
        runs, prev = [], None
        for n, aa in zip(g['residue_number'], g['aa']):
            if prev is None or n != prev + 1:
                runs.append('')
            runs[-1] += aa
            prev = n
        out[chain] = runs
    return out


@pytest.mark.parametrize('pdb', HELIX_SEQUENCES)
@pytest.mark.parametrize('method', mhc.HELIX_METHODS)
def test_helix_sequences_by_method(structures, pdb, method):
    assert _helix_sequences(structures[f'{pdb}/rcsb.pdb'], method) == \
        HELIX_SEQUENCES[pdb][method]


def test_helices_do_not_depend_on_the_file_numbering(structures):
    """A file with its own MHC numbering (here chain A shifted by 1, as in 7PBC or 5YXN):
    both methods find the same residues, while the Pierce residue numbers would be one
    residue off."""
    pf = structures['1AO7/rcsb.pdb']
    df = pf.df.drop(columns=['imgt', 'original_chain'])
    df['residue_number'] += (df['chain'] == 'A').to_numpy()
    shifted = TCRFrame(PDBFrame(df, name='1AO7 shifted'))
    for method in mhc.HELIX_METHODS:
        assert (mhc.mhc_helices(shifted, method=method) == mhc.mhc_helices(pf, method=method)).all()
    assert _helix_residues(shifted, 'imgt') == {('A', n + 1) for _, n in _helix_residues(pf, 'imgt')}
    assert _helix_residues(shifted, 'imgt') != _expand(mhc.PIERCE_HELICES['I'])


# ------------------------------------------------------- annotated structures

def test_annotate_numbers_the_mhc(structures):
    pf = structures['1AO7/rcsb.pdb']
    res = pf.df.drop_duplicates(['chain', 'observed_index'])
    a = res[res['chain'] == 'A'].set_index('residue_number')
    # HLA-A2 alpha1 numbering is IMGT numbering; alpha2 starts again at 1
    assert a.loc[50, 'imgt'] == '50' and a.loc[91, 'imgt'] == '1'
    assert set(a.loc[200:, 'imgt']) == {'-'}                     # alpha3: no G domain
    assert 'imgt_domain' not in pf.df
    num = mhc.mhc_numbering(pf)
    assert set(num['chain']) == {'A'} and len(num) == len(pf.sequences['A'])
    assert list(num['imgt']) == list(a['imgt'])
    assert num.loc[num['residue_number'] == 91, 'domain'].item() == 'G-ALPHA2'
    assert ''.join(num['aa']) == pf.sequences['A']
    assert set(num['topology']) == {'extracellular'}             # no signal peptide in the file


@pytest.mark.parametrize('path', ['3QIB/rcsb.pdb', '3QIB/af3-tbm.cif'])
def test_mouse_class_ii(structures, path):
    """3QIB is mouse I-E k: its alpha chain (EA) is numbered as G-ALPHA, its beta
    chain (EB) as G-BETA, in the experimental structure and in a model."""
    pf = structures[path]
    table = pf.chain_table
    assert table.loc[['A', 'B'], 'gene'].tolist() == ['EA', 'EB']
    assert table.loc[['A', 'B'], 'allele'].tolist() == ['H2-EAk', 'H2-EBk']
    num = mhc.mhc_numbering(pf)
    domains = num[num['domain'] != '-'].groupby('chain')['domain'].unique()
    assert domains.map(list).to_dict() == {'A': ['G-ALPHA'], 'B': ['G-BETA']}
    # each G domain is numbered all the way, strands (1-49) and helix
    for chain in 'AB':
        labels = num.loc[(num['chain'] == chain) & (num['domain'] != '-'), 'imgt']
        assert {'1', '49', '50', '72'} <= set(labels)


def test_regions_still_exports_the_helices():
    assert tcs.regions.mhc_helices is mhc.mhc_helices
    assert tcs.regions.PIERCE_HELICES is mhc.PIERCE_HELICES
