"""The defaults against an independent implementation (expected_defaults.csv, see
data/README.md).

Every column it has, in each of its three superpositions, on the 9 pairs it scores (it
leaves 3QIB out). Its docking angles are the 'v1' angles; the defaults are 'piercelab'.
"""

from pathlib import Path

import pandas as pd
import pytest

from tcrstruct import compare

DATA = Path(__file__).parent / 'data'
EXPECTED = pd.read_csv(DATA / 'expected_defaults.csv')
PAIRS = sorted(set(zip(EXPECTED['PDB'], EXPECTED['model'])))
MODES = ['full', 'pmhc', 'peptide_centric']

#: expected_defaults.csv column -> tcrstruct column
COLUMNS = {'Full_RMSD': 'Full', 'Peptide_RMSD': 'Peptide',
           'Peptide_Centric': 'Peptide_Centric', 'CDRA1': 'CDRA1', 'CDRA2': 'CDRA2',
           'CDRA3': 'CDRA3', 'CDRB1': 'CDRB1', 'CDRB2': 'CDRB2', 'CDRB3': 'CDRB3',
           'Mean_CDR': 'Mean_CDR', 'DockQ': 'DockQ', 'irms': 'iRMSD', 'lrmsd': 'LRMSD',
           'fnat': 'fnat', 'F1': 'F1', 'tm_score': 'tm_score', 'lddt': 'lddt',
           'alignment_identity': 'alignment_identity', 'TCR_LRMSD': 'TCR_LRMSD'}
ANGLES = ['crossing_angle', 'incident_angle', 'crossing_angle_error', 'incident_angle_error']

#: It fits Cα atoms in 'full' and 'pmhc', heavy atoms in 'peptide_centric'.
FIT = {'full': 'ca', 'pmhc': 'ca', 'peptide_centric': 'heavy'}


def _model_file(pdb, model):
    return next(p for p in (DATA / pdb).iterdir() if p.stem == model)


def _want(pdb, model, mode):
    e = EXPECTED
    return e[(e['PDB'] == pdb) & (e['model'] == model) & (e['Mode'] == mode)].iloc[0]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('pdb, model', PAIRS)
def test_defaults_match_expected(pdb, model, mode):
    kwargs = {} if FIT[mode] == 'ca' else {'align_atoms': 'heavy'}
    got = compare(_model_file(pdb, model), DATA / pdb / 'rcsb.pdb', name=pdb, align_on=mode,
                  **kwargs)
    want = _want(pdb, model, mode)
    for theirs, ours in COLUMNS.items():
        assert got[ours] == pytest.approx(want[theirs], rel=1e-4, abs=1e-6), ours


@pytest.mark.parametrize('pdb, model', PAIRS)
def test_v1_angles_match_expected(pdb, model):
    got = compare(_model_file(pdb, model), DATA / pdb / 'rcsb.pdb', name=pdb, angle_method='v1')
    want = _want(pdb, model, 'peptide_centric')
    for col in ANGLES:
        assert got[col] == pytest.approx(want[col], abs=1e-2), col
