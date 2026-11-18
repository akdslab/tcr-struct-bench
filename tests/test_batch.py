"""Directories, normalization and the command line."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from pdbframe import PDBFrame

import tcrstruct as tcs

from tcrstruct import (V1, TCRFrame, TCRStructWarning, compare, compare_dirs, normalize,
                       normalize_dir, pair_files)
from tcrstruct.batch import find_structures, key_pattern, pdb_key
from tcrstruct.cli import main
from tcrstruct.compare import REFERENCE_NAMES

DATA = Path(__file__).parent / 'data'
EXPECTED = pd.read_csv(DATA / 'expected_metrics.csv').rename(
    columns={theirs: ours for ours, theirs in REFERENCE_NAMES.items()})
EXPECTED['model'] = EXPECTED['input_file'].str.split('/').str[-3]

@pytest.fixture(scope='module')
def dirs(tmp_path_factory):
    """models/{method}/7T2D_model_*.{cif,pdb} plus one model without a reference."""
    root = tmp_path_factory.mktemp('bench')
    (root / 'models' / 'af3').mkdir(parents=True)
    (root / 'models' / 'boltz').mkdir()
    (root / 'refs').mkdir()
    shutil.copy(DATA / '7T2D' / 'af3-tbm.cif', root / 'models' / 'af3' / '7T2D_model_0.cif')
    shutil.copy(DATA / '7T2D' / 'boltz-2.pdb', root / 'models' / 'boltz' / '7t2d_model_1.pdb')
    shutil.copy(DATA / '6ZKW' / 'boltz-2.pdb', root / 'models' / 'boltz' / '6ZKW_orphan.pdb')
    shutil.copy(DATA / '7T2D' / 'rcsb.pdb', root / 'refs' / '7T2D.pdb')
    (root / 'refs' / '7T2D.fasta').write_text('>not a structure\n')
    return root


def _expected(model):
    return EXPECTED[(EXPECTED['PDB'] == '7T2D') & (EXPECTED['model'] == model)].iloc[0]


# ------------------------------------------------------------------- pairing

def test_keys():
    assert pdb_key('7n2n_model_3.cif.gz') == '7N2N'
    assert key_pattern(r'_(\w{4})_')(Path('af3_7n2n_seed1.cif')) == '7N2N'
    with pytest.raises(ValueError, match='does not match'):
        key_pattern(r'^\d{5}')('7n2n.pdb')


def test_find_and_pair(dirs):
    models = find_structures(dirs / 'models')
    assert [p.name for p in models] == ['7T2D_model_0.cif', '6ZKW_orphan.pdb',
                                        '7t2d_model_1.pdb']
    assert [p.name for p in find_structures(dirs / 'models', exclude=('boltz',))] == \
        ['7T2D_model_0.cif']
    pairs, unmatched = pair_files(models, find_structures(dirs / 'refs'))
    assert list(pairs['PDB']) == ['7T2D', '7T2D']
    assert [p.name for p in unmatched] == ['6ZKW_orphan.pdb']


def test_duplicate_references_are_refused(tmp_path):
    for name in ['7T2D.pdb', '7t2d_other.cif']:
        (tmp_path / name).write_text('')
    with pytest.raises(ValueError, match='both have key'):
        pair_files([], find_structures(tmp_path))


# ------------------------------------------------------------------- scoring

@pytest.mark.parametrize('n_jobs', [1, 2])
def test_compare_dirs(dirs, n_jobs):
    with pytest.warns(TCRStructWarning, match='no reference'):
        table = compare_dirs(dirs / 'models', dirs / 'refs', n_jobs=n_jobs, **V1)
    assert list(table['model']) == ['7T2D_model_0', '7t2d_model_1']
    assert list(table['method']) == ['af3', 'boltz']
    assert (table['error'] == '').all()
    assert [Path(p).name for p in table.attrs['unmatched']] == ['6ZKW_orphan.pdb']
    for (_, row), model in zip(table.iterrows(), ['af3-tbm', 'boltz-2']):
        assert row['DockQ'] == pytest.approx(_expected(model)['DockQ'], rel=1e-4)
        assert row['Full'] == pytest.approx(_expected(model)['Full'], rel=1e-4)
    # every row records the superposition it was scored in
    assert (table['align_on'] == 'peptide_centric').all()
    assert (table['align_atoms'] == 'heavy').all()
    assert table.attrs['versions'].startswith(f'tcrstruct {tcs.__version__}')
    assert table['DockQ'].dtype == float and table['peptide_reversed'].dtype == 'boolean'


def test_compare_dirs_resumes(dirs, tmp_path):
    """Pairs already in an earlier result are kept, not scored again."""
    out = tmp_path / 'metrics.csv'
    first = compare_dirs(dirs / 'models', dirs / 'refs', exclude=('boltz',), **V1)
    first.to_csv(out, index=False)
    with pytest.warns(TCRStructWarning, match='no reference'):
        table = compare_dirs(dirs / 'models', dirs / 'refs', existing=out, **V1)
    assert table.attrs['skipped'] == 1
    assert list(table['model']) == ['7T2D_model_0', '7t2d_model_1']
    assert (table['error'] == '').all()
    assert table['DockQ'].iat[0] == pytest.approx(first['DockQ'].iat[0], rel=1e-9)
    assert table['DockQ'].iat[1] == pytest.approx(_expected('boltz-2')['DockQ'], rel=1e-4)
    # a file that does not exist yet is an empty start
    with pytest.warns(TCRStructWarning, match='no reference'):
        fresh = compare_dirs(dirs / 'models', dirs / 'refs', existing=tmp_path / 'new.csv', **V1)
    assert fresh.attrs['skipped'] == 0 and len(fresh) == 2


# ---------------------------------------------------------------- normalizing

def test_normalize(tmp_path):
    out = normalize(DATA / '1AO7' / 'rcsb.pdb')
    assert out.chains == ['A', 'C', 'D', 'E', 'F']
    tcr = out.df[out.df['chain'].isin(['D', 'E'])]
    # TCR residue numbers are IMGT positions, insertions as insertion codes
    imgt = tcr['residue_number'].astype(str) + tcr['insertion_code']
    assert (imgt == tcr['imgt']).all()
    cys = tcr[tcr['residue_number'].isin([23, 104]) & (tcr['atom_name'] == 'SG')]
    assert len(cys) == 4
    # the written file reads back with the same numbering and scores the same
    path = out.write(tmp_path / '1ao7.pdb')
    back = PDBFrame.read(path)
    assert back.sequences == out.sequences
    assert back.df['residue_number'].tolist() == out.df['residue_number'].tolist()


def test_normalized_model_scores_the_same(tmp_path):
    ref = TCRFrame.read(DATA / '7T2D' / 'rcsb.pdb')
    raw = DATA / '7T2D' / 'boltz-2.pdb'
    path = normalize(raw, keep_leader=True).write(tmp_path / 'model.pdb')
    a, b = compare(raw, ref, name='7T2D'), compare(path, ref, name='7T2D')
    pd.testing.assert_series_equal(a.drop('PDB'), b.drop('PDB'), check_names=False,
                                   rtol=1e-5)


def test_normalize_dir(dirs, tmp_path):
    result = normalize_dir(dirs / 'models', tmp_path, fmt='pdb')
    assert (result['error'] == '').all()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.pdb')) == \
        ['af3/7T2D_model_0.pdb', 'boltz/6ZKW_orphan.pdb', 'boltz/7t2d_model_1.pdb']


# --------------------------------------------------------------- command line

def test_cli(dirs, tmp_path, capsys):
    out = tmp_path / 'metrics.csv'
    assert main(['compare-dirs', str(dirs / 'models'), str(dirs / 'refs'), '-o', str(out)]) == 0
    assert len(pd.read_csv(out)) == 2
    err = capsys.readouterr().err
    assert 'warning: 1 models have no reference' in err and '6ZKW_orphan.pdb' in err

    pair = [str(DATA / '7T2D' / 'boltz-2.pdb'), str(DATA / '7T2D' / 'rcsb.pdb')]
    assert main(['compare', *pair, '-o', str(tmp_path / 'one.csv'), '--v1']) == 0
    one = pd.read_csv(tmp_path / 'one.csv').iloc[0]
    assert one['DockQ'] == pytest.approx(_expected('boltz-2')['DockQ'], rel=1e-4)
    assert one['incident_angle'] == pytest.approx(_expected('boltz-2')['incident_angle'], abs=1e-2)
    # the defaults: DockQ v2 (see test_metric_versions.py)
    assert main(['compare', *pair, '-o', str(tmp_path / 'two.csv')]) == 0
    two = pd.read_csv(tmp_path / 'two.csv').iloc[0]
    assert two['DockQ'] == pytest.approx(0.873620, rel=1e-4)
    # 'v1' DockQ is measured in the superposition, so it needs the 'v1' fit too
    assert main(['compare', *pair, '-o', str(tmp_path / 'mixed.csv'), '--dockq', 'v1',
                 '--align-atoms', 'heavy']) == 0
    assert pd.read_csv(tmp_path / 'mixed.csv').iloc[0]['DockQ'] == pytest.approx(one['DockQ'])

    assert main(['normalize', str(DATA / '7T2D' / 'boltz-2.pdb'), str(tmp_path / 'n.cif')]) == 0
    assert PDBFrame.read(tmp_path / 'n.cif').chains == ['A', 'B', 'C', 'D', 'E']

    assert main(['chains', str(DATA / '7T2D' / 'boltz-2.pdb')]) == 0
    assert 'DPB1' in capsys.readouterr().out

    assert main(['info', str(DATA / '7T2D' / 'boltz-2.pdb')]) == 0
    out = capsys.readouterr().out
    assert 'MHC class II' in out and 'GLAWEWWRTVY' in out and 'crossing' in out
