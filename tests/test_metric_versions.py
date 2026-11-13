"""DockQ, TM-score and docking angles: the 'v2' / 'piercelab' defaults and the 'v1' versions.

The 'v1' numbers are checked against expected_metrics.csv in
test_benchmark.py; here, 'v2' against the programs it follows and the properties
each version should have.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pdbframe.structure.geometry import kabsch, transform

import tcrstruct as tcs
from tcrstruct import V1, TCRFrame, TCRStructWarning, compare
from tcrstruct.chains import scoring_frame
from tcrstruct.metrics import METRIC_METHODS, SUPERPOSITION_MODES, dockq, match, tm_score

DATA = Path(__file__).parent / 'data'
PDBS = ['1AO7', '6ZKW', '3QIB', '7T2D']
MODELS = ['af3-tbm', 'boltz-2', 'tcrmodel2']
CASES = [(p, m) for p in PDBS for m in MODELS]

# DockQ v2 itself on these pairs (data/README.md): TCR (D, E) and pMHC (A, B, C, F),
# each side one chain.
DOCKQ_V2 = pd.read_csv(DATA / 'expected_dockq_v2.csv').set_index(['PDB', 'model'])

# TM-score v2 (all matched Cα atoms on their own fit, L = matched residues); also
# checked against expected_defaults.csv in test_defaults.py.
TM_V2 = {('1AO7', 'af3-tbm'): 0.9905, ('1AO7', 'boltz-2'): 0.9934, ('1AO7', 'tcrmodel2'): 0.9580,
         ('6ZKW', 'af3-tbm'): 0.9862, ('6ZKW', 'boltz-2'): 0.9933, ('6ZKW', 'tcrmodel2'): 0.9734,
         ('3QIB', 'af3-tbm'): 0.9816, ('3QIB', 'boltz-2'): 0.9944, ('3QIB', 'tcrmodel2'): 0.9913,
         ('7T2D', 'af3-tbm'): 0.9025, ('7T2D', 'boltz-2'): 0.9818, ('7T2D', 'tcrmodel2'): 0.9743}


def _model_file(pdb, model):
    return next(p for p in (DATA / pdb).iterdir() if p.stem == model)


@pytest.fixture(scope='module')
def frames():
    """Scoring frames (standard chain IDs, heavy atoms) of every reference and model."""
    out = {}
    for pdb in PDBS:
        out[pdb] = scoring_frame(TCRFrame.read(DATA / pdb / 'rcsb.pdb', name=pdb))
        for m in MODELS:
            out[pdb, m] = scoring_frame(_model_file(pdb, m))
    return out


@pytest.fixture(scope='module')
def rows():
    """compare() with the defaults, for every pair."""
    return {(p, m): compare(_model_file(p, m), DATA / p / 'rcsb.pdb', name=p) for p, m in CASES}


def _moved(pf, seed=0):
    """`pf` turned and shifted by a random rigid motion."""
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R *= np.sign(np.linalg.det(R))
    df = pf.df.copy()
    df[['x', 'y', 'z']] = pf.coords @ R.T + rng.normal(scale=20, size=3)
    return type(pf)(df, name=pf.name)


# ---------------------------------------------------------------------- DockQ

@pytest.mark.parametrize('pdb, model', CASES)
def test_dockq_v2_matches_dockq(frames, pdb, model):
    got = dockq(frames[pdb], frames[pdb, model])
    want = DOCKQ_V2.loc[(pdb, model)]
    for col in ['nat_correct', 'nat_total', 'nonnat_count', 'model_total', 'clashes', 'len1', 'len2']:
        assert got[col] == want[col], col
    for col in ['DockQ', 'fnat', 'F1', 'iRMSD', 'LRMSD']:
        assert got[col] == pytest.approx(want[col], rel=1e-4), col


def test_dockq_v2_is_the_default(frames, rows):
    row = rows['1AO7', 'af3-tbm']
    assert row['DockQ'] == pytest.approx(DOCKQ_V2.loc[('1AO7', 'af3-tbm'), 'DockQ'], rel=1e-4)
    assert dockq(frames['1AO7'], frames['1AO7', 'af3-tbm'], method='v2')['DockQ'] == \
        dockq(frames['1AO7'], frames['1AO7', 'af3-tbm'])['DockQ']


def test_dockq_v2_superposes_by_itself(frames):
    ref, model = frames['7T2D'], frames['7T2D', 'af3-tbm']
    a, b = dockq(ref, model), dockq(ref, _moved(model))
    for col in ['DockQ', 'iRMSD', 'LRMSD', 'fnat', 'clashes']:
        assert b[col] == pytest.approx(a[col], rel=1e-9, abs=1e-9), col
    # ... and pairs residues by itself: matched frames give the same score
    r, m, _ = match(ref, model)
    assert dockq(r, m)['DockQ'] == pytest.approx(a['DockQ'], rel=1e-4)


def test_dockq_v2_of_a_structure_with_itself(frames):
    d = dockq(frames['3QIB'], frames['3QIB'])
    assert d['DockQ'] == pytest.approx(1.0)
    assert d['fnat'] == 1 and d['nonnat_count'] == 0
    assert d['iRMSD'] == pytest.approx(0, abs=1e-6) and d['LRMSD'] == pytest.approx(0, abs=1e-6)


def test_dockq_v1_is_the_reference_benchmarks(frames):
    """'v1': peptide core (9 residues) against MHC and CDRs, in the current pose."""
    ref, model = frames['1AO7'], frames['1AO7', 'af3-tbm']
    r, m, _ = match(ref, model)
    m = tcs.metrics.superpose(r, m, atoms='heavy')
    v1 = dockq(r, m, method='v1')
    assert v1['len1'] == 9                      # the peptide core is the ligand
    assert v1['DockQ'] == pytest.approx(0.961, abs=1e-3)
    # no refit: moving the model changes v1, not v2
    assert dockq(r, _moved(m), method='v1')['LRMSD'] > 10
    assert dockq(r, _moved(m))['LRMSD'] == pytest.approx(dockq(r, m)['LRMSD'], rel=1e-9)


# ------------------------------------------------------------------- TM-score

@pytest.mark.parametrize('pdb, model', CASES)
def test_tm_score_v2(rows, pdb, model):
    assert rows[pdb, model]['tm_score'] == pytest.approx(TM_V2[pdb, model], abs=1e-4)


def test_tm_score_v2_fits_by_itself(frames):
    ref, model = frames['6ZKW'], frames['6ZKW', 'af3-tbm']
    r, m, _ = match(ref, model)
    n = len(r.df[r.df['atom_name'] == 'CA'].drop_duplicates(['chain', 'observed_index']))
    v2 = tm_score(r, m)
    assert tm_score(r, _moved(m)) == pytest.approx(v2, rel=1e-9)     # pose does not matter
    # it is the current-pose score after a Kabsch fit of all Cα atoms
    ca = (r.df['atom_name'] == 'CA').to_numpy()
    fitted = m.transform(*kabsch(m.coords[ca], r.coords[ca]))
    assert v2 == pytest.approx(tm_score(r, fitted, method='v1'), rel=1e-9)
    assert tm_score(r, r) == pytest.approx(1.0)
    # a longer L (say the whole reference) counts missing residues against the model
    assert tm_score(r, m, length=2 * n) < 0.6 * v2


def test_tm_score_v1_is_the_current_pose(frames):
    r, m, _ = match(frames['1AO7'], frames['1AO7', 'af3-tbm'])
    m = tcs.metrics.superpose(r, m, atoms='heavy')
    assert tm_score(r, m, method='v1') == pytest.approx(0.9867, abs=1e-4)
    assert tm_score(r, _moved(m), method='v1') < 0.1


# -------------------------------------------------------------- compare() and V1

def test_v2_scores_do_not_depend_on_the_superposition_mode():
    pdb, model = '3QIB', 'tcrmodel2'
    rows = [compare(_model_file(pdb, model), DATA / pdb / 'rcsb.pdb', align_on=m)
            for m in SUPERPOSITION_MODES]
    for col in ['DockQ', 'iRMSD', 'LRMSD', 'fnat', 'tm_score']:
        assert len({round(float(r[col]), 9) for r in rows}) == 1, col


def test_v1_sets_every_metric_to_the_reference_benchmarks(rows):
    assert V1 == {'align_atoms': 'heavy', 'angle_method': 'v1', 'dockq_method': 'v1',
                  'tm_method': 'v1'}
    pdb, model = '1AO7', 'af3-tbm'
    v1 = compare(_model_file(pdb, model), DATA / pdb / 'rcsb.pdb', name=pdb, **V1)
    default = rows[pdb, model]
    assert v1['DockQ'] == pytest.approx(0.961, abs=1e-3) and v1['len1'] == 9
    assert default['DockQ'] == pytest.approx(0.9096, abs=1e-4) and default['len1'] == 324
    assert v1['tm_score'] == pytest.approx(0.9867, abs=1e-4)
    assert default['tm_score'] == pytest.approx(0.9905, abs=1e-4)
    # piercelab angles by default, the 'v1' ones with V1
    assert default['incident_angle'] == pytest.approx(10.3303, abs=1e-3)
    assert v1['incident_angle'] == pytest.approx(10.7124, abs=1e-3)
    # the fit: heavy atoms (V1) or Cα (default); lDDT has one version
    assert v1['Full'] == pytest.approx(1.4945, abs=1e-4)
    assert default['Full'] == pytest.approx(1.4920, abs=1e-4)
    assert v1['lddt'] == default['lddt']


# The implementation of expected_defaults.csv (data/README.md) on 1AO7 af3-tbm: its
# 'full' and 'pmhc' modes fit Cα atoms, its 'peptide_centric' mode heavy atoms.
EXPECTED_FULL_RMSD = {('full', 'ca'): 1.3776393, ('pmhc', 'ca'): 1.5924403,
                 ('peptide_centric', 'heavy'): 1.4945355}


@pytest.mark.parametrize('mode, fit', EXPECTED_FULL_RMSD)
def test_fit_atoms_match_expected(mode, fit):
    row = compare(_model_file('1AO7', 'af3-tbm'), DATA / '1AO7' / 'rcsb.pdb', align_on=mode,
                  align_atoms=fit)
    assert row['Full'] == pytest.approx(EXPECTED_FULL_RMSD[mode, fit], rel=1e-6)


def test_default_fit_is_on_c_alpha(frames):
    r, m, _ = match(frames['6ZKW'], frames['6ZKW', 'boltz-2'])
    ca = (r.df['atom_name'] == 'CA').to_numpy()
    R, t = tcs.metrics.superposition(r, m, mode='full')
    R0, t0 = kabsch(m.coords[ca], r.coords[ca])
    assert np.allclose(R, R0) and np.allclose(t, t0)
    R1, _ = tcs.metrics.superposition(r, m, mode='full', atoms='heavy')
    assert not np.allclose(R, R1)
    with pytest.raises(ValueError, match='atoms must be one of'):
        tcs.metrics.superposition(r, m, atoms='backbone')


@pytest.mark.parametrize('arg', ['dockq_method', 'tm_method', 'align_atoms'])
def test_metric_method_must_be_known(arg):
    assert METRIC_METHODS == ('v1', 'v2')
    with pytest.raises(ValueError, match=f'{arg} must be one of'):
        compare(DATA / '1AO7' / 'af3-tbm.cif', DATA / '1AO7' / 'rcsb.pdb', **{arg: 'v3'})
