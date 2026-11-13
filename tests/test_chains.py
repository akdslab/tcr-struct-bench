"""detect_chains: which chain of a complex is which, and its standard ID."""

from pathlib import Path

import pandas as pd
import pytest
from pdbframe import PDBFrame
from tcrkit.mhc import mhc_database

import tcrstruct as tcs

DATA = Path(__file__).parent / 'data'

# HLA-A*02:01 alpha1 + alpha2 (PDB 1AO7 chain A)
HLA_A2 = ('GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLG'
          'TLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLR'
          'AYLEGTCVEWLRRYLENGKETLQRT')
DRA = ('IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSN'
       'YTPITN')
DRB1 = ('GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYC'
        'RHNYGVGESFTVQRR')
B2M = ('IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYA'
       'CRVNHVTLSQPKIVKWDRDM')
BETA = ('MNAGVTQTPKFRVLKTGQSMTLLCAQDMNHDYMYWYRQDPGMGLRLIHYSVGEGTTAKGEVPDGYNVSR'
        'LKKQNFLLGLESAAPSQTSVYFCASSFTDTQYFGPGTRLTVL')
ALPHA = ('AQKITQTQPGMFVQEKEAVTLDCTYDTSDPSYGLFWYKQPSSGEMIFLIYQGSYDQQNATEGRYSLNFQ'
         'KARKSANLVISASQLGDSAMYFCAMSQLNSGNTPLVFGKGTRLSVIA')


def _mouse(allele):
    db = mhc_database()
    return db.loc[db['allele'] == allele, 'sequence'].iloc[0]


@pytest.fixture(scope='module')
def class_i():
    # file chain IDs that are not the standard ones
    return tcs.detect_chains({'H': HLA_A2, 'L': B2M, 'P': 'LLFGYPVYV', 'Q': ALPHA, 'R': BETA})


def test_class_i(class_i):
    assert list(class_i.columns) == ['type', 'details', 'new_chain', 'sequence']
    assert class_i['new_chain'].to_dict() == {'H': 'A', 'L': 'F', 'P': 'C', 'Q': 'D', 'R': 'E'}
    assert class_i['type'].to_dict() == {'H': 'MHC', 'L': 'MHC', 'P': 'peptide', 'Q': 'TCR',
                                         'R': 'TCR'}
    # ANARCI, with every chain type allowed, calls this alpha chain delta ('D');
    # a D chain is taken as alpha when there is no A
    assert class_i.loc[['Q', 'R'], 'details'].tolist() == ['D', 'Beta']
    assert class_i.loc['H', 'details'] == 'A'


def test_class_ii_and_extra_chains():
    t = tcs.detect_chains({'A': DRA, 'B': DRB1, 'C': 'PKYVKQNTLKLAT', 'X': 'HOH'},
                          numbering={})
    assert t['new_chain'].to_dict() == {'A': 'A', 'B': 'B', 'C': 'C', 'X': 'G'}
    assert t.loc['X', 'type'] == 'None'


def test_sequences_as_a_series(class_i):
    seqs = pd.Series({'R': BETA, 'Q': ALPHA, 'P': 'LLFGYPVYV', 'L': B2M, 'H': HLA_A2})
    pd.testing.assert_frame_equal(tcs.detect_chains(seqs), class_i)
    with pytest.raises(TypeError, match='chain ID: sequence'):
        tcs.detect_chains([HLA_A2, B2M])


def test_mouse_chains():
    # I-A beta (AB) first: its name has an A in it too, which once made it an alpha
    t = tcs.detect_chains({'A': _mouse('H2-ABb'), 'B': _mouse('H2-AAb'),
                           'C': 'GELIGILNAAKVPAD'}, numbering={})
    assert list(t['details']) == ['AB', 'AA', '-']
    assert list(t['new_chain']) == ['B', 'A', 'C']
    # a class I heavy chain without B2M
    t = tcs.detect_chains({'H': _mouse('H2-Kb'), 'P': 'SIINFEKL'}, numbering={})
    assert list(t['new_chain']) == ['A', 'C']


def test_one_chain_of_class_ii_is_not_labelled():
    t = tcs.detect_chains({'A': DRA, 'C': 'PKYVKQNTLKLAT'}, numbering={})
    assert t['new_chain'].to_dict() == {'A': 'G', 'C': 'C'}


@pytest.mark.parametrize('pdb, ids', [('1AO7', 'ACDEF'), ('3QIB', 'ABCDE')])
def test_structure(pdb, ids):
    pf = PDBFrame.read(DATA / pdb / 'rcsb.pdb')
    table = tcs.detect_chains(pf)
    pd.testing.assert_frame_equal(table, tcs.detect_chains(pf.sequences))
    assert set(ids) <= set(table['new_chain'])
