"""The 'v1' scores must match expected_metrics.csv for the pairs in tests/data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pdbframe import PDBFrame

from tcrstruct import COLUMNS, V1, TCRFrame, compare, read_processed
from tcrstruct.chains import scoring_frame
from tcrstruct.compare import REFERENCE_NAMES
from tcrstruct.numbering import Numbering, imgt_labels
from tcrstruct.metrics import SUPERPOSITION_MODES, lddt_coords, match, superposition

DATA = Path(__file__).parent / 'data'
EXPECTED = pd.read_csv(DATA / 'expected_metrics.csv').rename(
    columns={theirs: ours for ours, theirs in REFERENCE_NAMES.items()})
EXPECTED['model'] = EXPECTED['input_file'].str.split('/').str[-3]
CASES = list(EXPECTED[['PDB', 'model']].itertuples(index=False, name=None))

# Every metric is its 'v1' version (**V1; the defaults, angles
# 'piercelab' and DockQ / TM-score 'v2', are tested in test_tcr.py and
# test_metric_versions.py). The expected angles carry float32 noise: the TCR
# alpha/beta rotation is close to 180 degrees, where its axis is ill-conditioned.
ANGLE_COLS = ['crossing_angle', 'incident_angle', 'crossing_angle_error', 'incident_angle_error']
VALUE_COLS = [c for c in COLUMNS if c != 'PDB']

def _model_file(pdb, model):
    return next(p for p in (DATA / pdb).iterdir() if p.stem == model)


@pytest.fixture(scope='module')
def references():
    return {pdb: TCRFrame.read(DATA / pdb / 'rcsb.pdb', name=pdb)
            for pdb in EXPECTED['PDB'].unique()}


def _check(row: pd.Series, pdb: str, model: str):
    want = EXPECTED[(EXPECTED['PDB'] == pdb) & (EXPECTED['model'] == model)].iloc[0]
    got, want = row[VALUE_COLS].astype(float), want[VALUE_COLS].astype(float)
    assert got.isna().equals(want.isna())
    for col in VALUE_COLS:
        if np.isnan(want[col]):
            continue
        tol = dict(abs=1e-2) if col in ANGLE_COLS else dict(rel=1e-4, abs=1e-6)
        assert got[col] == pytest.approx(want[col], **tol), col


# ------------------------------------------------------------- full benchmark

@pytest.mark.parametrize('pdb, model', CASES)
def test_compare_raw_files(references, pdb, model):
    """From the raw prediction files: chain detection, ANARCI and every metric."""
    row = compare(_model_file(pdb, model), references[pdb], name=pdb, **V1)
    _check(row, pdb, model)


@pytest.mark.parametrize('pdb, model', CASES)
def test_compare_reference_cache(references, pdb, model):
    """From the prepared predictions the expected values were scored from (*_processed.csv)."""
    row = compare(read_processed(DATA / pdb / f'{model}_processed.csv'),
                  references[pdb], name=pdb, **V1)
    _check(row, pdb, model)


# ---------------------------------------------------------------- preparation

@pytest.mark.parametrize('pdb, model', CASES)
def test_annotation_matches_reference_cache(pdb, model):
    """Same chains, atoms, sequences and TCR IMGT labels as the prepared predictions."""
    mine = scoring_frame(_model_file(pdb, model))
    theirs = read_processed(DATA / pdb / f'{model}_processed.csv')
    assert mine.sequences == theirs.sequences
    assert len(mine) == len(theirs)
    tcr = mine.df['chain'].isin(['D', 'E'])
    labels = mine.df.loc[tcr, 'imgt'].to_numpy(), theirs.df.loc[tcr, 'imgt'].to_numpy()
    if pdb == '3QIB' and model == 'boltz-2':
        # an older cache: residues before the V domain are all '-1' there
        v = ~pd.Series(labels[1]).eq('-1').to_numpy()
        labels = labels[0][v], labels[1][v]
    assert (labels[0] == labels[1]).all()


# ---------------------------------------------------------------- unit checks

def test_imgt_labels():
    num = Numbering(labels=['1', '2', '111A', '128'], residues='ABCD', query_start=2,
                    chain_type='A')
    assert imgt_labels(8, num) == ['-2', '-1', '1', '2', '111A', '128', '129', '130']
    assert imgt_labels(3, None) == ['-', '-', '-']
    with pytest.raises(ValueError, match='covers'):
        imgt_labels(4, num)


def test_lddt_of_identical_coordinates_is_one():
    rng = np.random.default_rng(0)
    xyz = rng.normal(scale=5, size=(40, 3))
    residue = np.repeat(np.arange(10), 4)
    is_ca = np.tile([False, True, False, False], 10)
    assert lddt_coords(xyz, xyz, residue, is_ca) == pytest.approx(1.0)


def test_self_comparison(references):
    ref = references['1AO7']
    row = compare(ref, ref, name='1AO7')
    assert row['Full'] == pytest.approx(0, abs=1e-6)
    assert row['DockQ'] == pytest.approx(1.0)
    assert row['lddt'] == pytest.approx(1.0)
    assert row['tm_score'] == pytest.approx(1.0)
    assert row['crossing_angle_error'] == pytest.approx(0, abs=1e-4)


def test_match_needs_a_common_chain():
    a = PDBFrame(pd.DataFrame({'chain': 'A', 'residue_number': [1], 'residue_name': 'GLY',
                               'atom_name': 'CA', 'x': 0.0, 'y': 0.0, 'z': 0.0}))
    b = a.rename_chains({'A': 'B'})
    with pytest.raises(ValueError, match='no chains'):
        match(a, b)


# ------------------------------------------------------------ superposition

@pytest.mark.parametrize('pdb, model', [('1AO7', 'af3-tbm'), ('3QIB', 'tcrmodel2')])
def test_superposition_modes(references, pdb, model):
    # heavy-atom fits, so each fit minimises the (heavy-atom) RMSD of what it fits on
    rows = {m: compare(_model_file(pdb, model), references[pdb], name=pdb, align_on=m,
                       align_atoms='heavy') for m in SUPERPOSITION_MODES}
    assert rows['full']['Full'] <= min(r['Full'] for r in rows.values()) + 1e-9
    assert rows['peptide_centric']['Peptide_Centric'] <= \
        min(r['Peptide_Centric'] for r in rows.values()) + 1e-9
    # superposition-free scores do not move
    for col in ['lddt', 'nat_total', 'nat_correct', 'crossing_angle', 'alignment_identity']:
        assert len({round(float(r[col]), 9) for r in rows.values()}) == 1, col
    with pytest.raises(ValueError, match='must be one of'):
        compare(_model_file(pdb, model), references[pdb], align_on='tcr_c')


@pytest.mark.parametrize('pdb, model', [('1AO7', 'af3-tbm'), ('3QIB', 'tcrmodel2')])
@pytest.mark.parametrize('atoms', ['heavy', 'ca'])
def test_each_mode_fits_its_own_atoms_best(references, pdb, model, atoms):
    """Fitted on heavy atoms, or on their Cα atoms (the default)."""
    ref, mod, _ = match(scoring_frame(references[pdb]),
                        scoring_frame(_model_file(pdb, model)))
    fits = {m: superposition(ref, mod, m, atoms=atoms) for m in SUPERPOSITION_MODES}
    masks = TCRFrame(ref).masks
    picked = {'tcr': ref.df['chain'].isin(['D', 'E']).to_numpy(),
              'v_domains': masks['Va'] | masks['Vb'],
              'cdrs': np.logical_or.reduce([masks[c] for c in
                                            ['CDRA1', 'CDRA2', 'CDRA3', 'CDRB1', 'CDRB2', 'CDRB3']])}
    if atoms == 'ca':
        ca = (ref.df['atom_name'] == 'CA').to_numpy()
        picked = {m: atoms & ca for m, atoms in picked.items()}
    for mode, atoms in picked.items():
        def fit_rmsd(m):
            moved = mod.transform(*fits[m]).coords[atoms]
            return np.sqrt(((moved - ref.coords[atoms]) ** 2).sum(1).mean())
        assert fit_rmsd(mode) <= min(fit_rmsd(m) for m in fits) + 1e-9, mode



def test_rows_record_the_superposition():
    """Each row ends with the superposition it was scored in and the package versions."""
    import tcrstruct as tcs
    model, ref = DATA / '3QIB' / 'tcrmodel2.pdb', DATA / '3QIB' / 'rcsb.pdb'
    row = compare(model, ref, align_on='pmhc')
    assert list(row.index) == [*COLUMNS, *tcs.NEW_COLUMNS, *tcs.SETTINGS]
    assert row[['align_on', 'align_atoms']].tolist() == ['pmhc', 'ca']
    assert row.attrs['versions'].startswith(f'tcrstruct {tcs.__version__}, pdbframe ')
    assert compare(model, ref, **V1)['align_atoms'] == 'heavy'
