"""The scoring helpers: metric groups, docking RMSDs, peptide direction, per-residue lDDT,
compare_models, and the CLI settings file."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pdbframe import PDBFrame

import tcrstruct as tcs
from tcrstruct import TCRFrame, compare
from tcrstruct.angles import peptide_direction
from tcrstruct.cli import main
from tcrstruct.regions import CDRS

DATA = Path(__file__).parent / 'data'
MODEL, REF = DATA / '1AO7' / 'af3-tbm.cif', DATA / '1AO7' / 'rcsb.pdb'


@pytest.fixture(scope='module')
def ref():
    return TCRFrame.read(REF, name='1AO7')


def test_metric_groups_cover_every_column():
    grouped = {c for cols in tcs.METRIC_GROUPS.values() for c in cols}
    assert grouped >= set(tcs.COLUMNS) | set(tcs.NEW_COLUMNS) | set(tcs.SETTINGS)
    assert 'main' not in tcs.METRIC_GROUPS


def test_grouped_output(ref):
    row = tcs.compare(DATA / '1AO7' / 'af3-tbm.cif', ref, grouped=True)
    assert list(row.index.names) == ['group', 'column']
    assert list(dict.fromkeys(row.index.get_level_values('group'))) == [
        'pair', 'global', 'peptide', 'cdrs', 'docking', 'angles', 'contacts', 'bookkeeping']
    assert list(row['global'].index) == ['Full', 'lddt', 'tm_score']
    assert row.attrs['versions'].startswith('tcrstruct ')
    flat = tcs.compare(DATA / '1AO7' / 'af3-tbm.cif', ref)
    assert row[('global', 'Full')] == flat['Full']
    table = tcs.compare_models({'af3': DATA / '1AO7' / 'af3-tbm.cif'}, ref, grouped=True)
    assert table['cdrs'].columns.tolist() == [*tcs.METRIC_GROUPS['cdrs']]
    assert list(row['angles'].index) == ['crossing_angle', 'incident_angle',
                                         'crossing_angle_error', 'incident_angle_error']


def test_several_superpositions(ref):
    modes = ['peptide_centric', 'pmhc', 'cdrs']
    table = compare(MODEL, ref, align_on=modes)
    assert list(table.index) == modes and list(table['align_on']) == modes
    for mode in modes:     # each row is the single-superposition row
        one = compare(MODEL, ref, align_on=mode)
        pd.testing.assert_series_equal(table.loc[mode], one, check_names=False,
                                       check_dtype=False)
    # superposition-free metrics are the same in every row, RMSDs are not
    assert table['lddt'].nunique() == 1 and table['Full'].nunique() == 3
    models = tcs.compare_models({'af3': MODEL}, ref, align_on=['full', 'tcr'])
    assert list(models.index) == [('af3', 'full'), ('af3', 'tcr')]
    pairs = pd.DataFrame({'PDB': ['1AO7'] * 2, 'model_file': [MODEL, DATA / 'missing.cif'],
                          'reference_file': [REF] * 2})
    rows = tcs.compare_pairs(pairs, align_on=['full', 'tcr'])
    assert list(rows['align_on']) == ['full', 'tcr'] * 2
    assert list(rows['error'] == '') == [True, True, False, False]
    with pytest.raises(ValueError, match='align_on'):
        compare(MODEL, ref, align_on=['full', 'nowhere'])


def test_a_cdr_gap_is_numbered_on_the_deposited_sequence():
    path = DATA / '9RUP' / 'rcsb.cif'
    with pytest.warns(tcs.TCRStructWarning):
        c1, c2 = (TCRFrame.read(path, complex=n) for n in (1, 2))
    # copy 2 has no atoms for residues 52-65 of its TCR beta, next to CDRB2; numbered on
    # the deposited sequence (unobserved residues included) its CDRB2 is copy 1's
    res = c2.residues
    gap = res[(res['chain'] == 'E') & res['residue_number'].between(52, 65)]
    assert not gap['observed'].any() and (gap['atoms'] == 0).all()
    assert gap['imgt'].iloc[0] == '65' and gap['region'].iloc[0] == 'CDRB2'
    assert c1.cdrs['CDRB2'] == 'YFSETQ' and c2.cdrs['CDRB2'] == 'YFSET'   # Q has no atoms
    assert c2.select(regions='cdrb2').unobserved['aa'].tolist() == ['Q']
    row = compare(c2, c1)
    assert not row[[*CDRS, 'Mean_CDR']].isna().any()
    # numbered on the residues with atoms only, the labels after the gap shift: CDRB2 is
    # 'YFS', cannot be paired with copy 1's, and only CDRB2 and Mean_CDR are blank
    with pytest.warns(tcs.TCRStructWarning):
        plain = PDBFrame.read(path)
        shifted = TCRFrame(PDBFrame(plain.df, name=plain.name), complex=2)   # atoms only
    assert shifted.cdrs['CDRB2'] == 'YFS'
    with pytest.warns(tcs.TCRStructWarning, match='CDRB2 RMSD not computed'):
        gapped = compare(shifted, c1)
    assert gapped[['CDRB2', 'Mean_CDR']].isna().all()
    assert gapped[['Full', 'Peptide', 'CDRA1', 'CDRB3']].notna().all()
    assert gapped['CDRA1'] == pytest.approx(row['CDRA1'], rel=1e-4)


def test_unobserved_residues_follow_the_frame(tmp_path):
    path = DATA / '9RUP' / 'rcsb.cif'
    with pytest.warns(tcs.TCRStructWarning):
        c2 = TCRFrame.read(path, complex=2)
    unseen = c2.unobserved
    assert set(unseen['chain']) == {'A', 'D', 'E'}         # standard chain IDs
    assert (c2.residues.loc[~c2.residues['observed'], 'original_chain'] != '').all()
    # selections: by chain all of them, by region those the labels place there, by
    # contacts (a mask) the runs inside what is kept
    assert len(c2.select(chains='E').unobserved) == (unseen['chain'] == 'E').sum()
    vb = c2.select(regions='Vb').unobserved
    assert set(vb['chain']) == {'E'} and vb['residue_number'].between(52, 65).all()
    assert len(c2.select(regions='peptide_core').unobserved) == 0
    assert len(c2.select(chains='E', invert=True).unobserved.query("chain == 'E'")) == 0
    assert len(c2.select(atoms='ca').unobserved) == len(unseen)
    # normalize numbers them by IMGT too, and mmCIF keeps them
    with pytest.warns(tcs.TCRStructWarning):
        norm = tcs.normalize(path, complex=2)
    gap = norm.residues.query("chain == 'E' and not observed")
    assert (gap['residue_number'].astype(str) + gap['insertion_code'] == gap['imgt']).all()
    norm.write(tmp_path / 'norm.cif')
    back = TCRFrame.read(tmp_path / 'norm.cif')
    cols = ['chain', 'residue_number', 'insertion_code', 'aa', 'observed', 'imgt', 'region']
    assert back.residues[cols].equals(norm.residues[cols])
    with pytest.warns(UserWarning, match='write .cif'):
        norm.write(tmp_path / 'norm.pdb')
    # each copy found by contacts keeps its own
    copies = tcs.find_complexes(path)
    assert copies[1].residues[cols].equals(c2.residues[cols])


def test_docking_rmsds_follow_the_superposition(ref):
    on_pmhc = compare(MODEL, ref, align_on='pmhc')
    on_tcr = compare(MODEL, ref, align_on='tcr')
    # fitted on a side, that side's own RMSD is the smallest of the two poses
    assert on_pmhc['pMHC_LRMSD'] < on_tcr['pMHC_LRMSD']
    assert on_tcr['TCR_LRMSD'] < on_pmhc['TCR_LRMSD']
    # the pMHC-aligned L-RMSD of expected_defaults.csv (test_defaults.py checks every pair)
    assert on_pmhc['TCR_LRMSD'] == pytest.approx(1.616985, rel=1e-4)


def _reversed_peptide(frame: TCRFrame) -> TCRFrame:
    """`frame` with its peptide turned 180 degrees about the groove normal."""
    v = frame.docking_vectors()
    n = v['groove_normal']
    R = 2 * np.outer(n, n) - np.eye(3)             # 180 degrees about n
    df = frame.df.copy()
    pep = (df['chain'] == 'C').to_numpy()
    xyz = df.loc[pep, ['x', 'y', 'z']].to_numpy(float)
    center = xyz.mean(axis=0)
    moved = ((xyz - center) @ R.T + center).astype(df['x'].dtype)
    df.loc[pep, ['x', 'y', 'z']] = moved
    return TCRFrame(frame._same(df))


def test_peptide_reversed(ref):
    assert peptide_direction(ref) > 0.8
    backwards = _reversed_peptide(ref)
    assert peptide_direction(backwards) < -0.8
    assert not compare(MODEL, ref)['peptide_reversed']
    assert compare(backwards, ref)['peptide_reversed']


def test_per_residue_lddt(ref):
    res = tcs.per_residue_lddt(MODEL, ref)
    assert {'chain', 'imgt', 'region', 'lddt'} <= set(res.columns)
    assert res['lddt'].mean() == pytest.approx(compare(MODEL, ref)['lddt'], rel=1e-9)
    by_region = tcs.per_residue_lddt(MODEL, ref, by='region')
    assert 'CDRB3' in by_region.index and by_region.is_monotonic_increasing
    with pytest.raises(ValueError, match='by must be'):
        tcs.per_residue_lddt(MODEL, ref, by='atom')


def test_compare_models(ref):
    table = tcs.compare_models({'af3': MODEL, 'boltz': DATA / '1AO7' / 'boltz-2.pdb'}, ref)
    assert list(table.index) == ['af3', 'boltz']
    assert table['DockQ'].dtype == float
    assert table.attrs['versions'].startswith(f'tcrstruct {tcs.__version__}')
    single = compare(MODEL, ref)
    assert table.loc['af3', 'Full'] == pytest.approx(single['Full'], rel=1e-12)
    assert list(tcs.compare_models([MODEL], ref).index) == ['af3-tbm']


def test_cli_writes_settings(tmp_path):
    out = tmp_path / 'row.csv'
    assert main(['-q', 'compare', str(MODEL), str(REF), '-o', str(out), '--align-on', 'pmhc']) == 0
    row = pd.read_csv(out).iloc[0]
    assert row['align_on'] == 'pmhc' and 'versions' not in row
    settings = json.loads(out.with_suffix('.settings.json').read_text())
    assert settings['align_on'] == 'pmhc' and settings['dockq_method'] == 'v2'
    assert settings['versions'].startswith('tcrstruct ')


def test_example_data():
    assert tcs.example() == DATA.resolve()
    assert tcs.example('1AO7', 'rcsb.pdb').is_file()
    with pytest.raises(FileNotFoundError, match='the examples are .*1AO7'):
        tcs.example('XXXX', 'rcsb.pdb')
