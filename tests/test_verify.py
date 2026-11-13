"""verify, fused peptides, and the database sources."""

import urllib.request
from pathlib import Path

import pandas as pd
import pytest

import tcrstruct as tcs
from tcrstruct import TCRFrame
from tcrstruct.cli import main

DATA = Path(__file__).parent / 'data'


def _online() -> bool:
    try:
        urllib.request.urlopen('https://files.rcsb.org', timeout=10)
        return True
    except Exception:
        return False


needs_network = pytest.mark.skipif(not _online(), reason='no network')


def _status(table) -> dict:
    return dict(zip(table['check'], table['status']))


# --------------------------------------------------------------------- verify

def test_complete_complexes_pass():
    for pdb in ['1AO7', '3QIB']:
        v = tcs.verify(DATA / pdb / 'rcsb.pdb')
        assert v.attrs['valid'], v
        assert set(_status(v)) >= {'chains', 'peptide in the MHC groove', 'TCR on the pMHC',
                                   'TCR alpha-beta pairing'}


def test_missing_b2m():
    pmhc = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').select(chains=['A', 'C'])
    strict = tcs.verify(pmhc, kind='pmhc', require_b2m=True)
    assert not strict.attrs['valid'] and 'B2M' in strict.set_index('check').at['chains', 'detail']
    loose = tcs.verify(pmhc, kind='pmhc')
    assert loose.attrs['valid'] and _status(loose)['chains'] == 'warning'
    assert not tcs.is_valid(pmhc)            # a TCR-pMHC needs the TCR


def test_sequences_only():
    seqs = TCRFrame.read(DATA / '7T2D' / 'boltz-2.pdb').sequences
    v = tcs.verify(seqs)
    assert v.attrs['valid'] and 'peptide in the MHC groove' not in set(v['check'])


def test_verify_dir(tmp_path):
    for pdb in ['1AO7', '6XCP']:
        src = next((DATA / pdb).glob('rcsb.*'))
        (tmp_path / f'{pdb}{src.suffix}').write_bytes(src.read_bytes())
    table = tcs.verify_dir(tmp_path)
    assert dict(zip(table['file'].map(lambda f: Path(f).stem), table['valid'])) == \
        {'1AO7': True, '6XCP': False}
    assert 'fused to chain B' in table.loc[~table['valid'], 'failed'].iloc[0]


# --------------------------------------------------------------- fused peptide

def test_find_fused_peptide_6xcp():
    found = tcs.find_fused_peptide(DATA / '6XCP' / 'rcsb.cif')
    assert len(found) == 1
    hit = found.iloc[0]
    assert (hit['chain'], hit['side'], hit['start'], hit['end']) == ('B', 'N', '-2', '14')
    assert hit['peptide'] == 'GQVELGGGNAVEVCKGS' and hit['evidence'] == 'structure'


def test_no_fused_peptide_elsewhere():
    for f in [DATA / '1AO7' / 'rcsb.pdb', DATA / '3QIB' / 'af3-tbm.cif']:
        assert tcs.find_fused_peptide(f).empty


def test_fused_peptide_from_sequence():
    seqs = TCRFrame.read(DATA / '6XCP' / 'rcsb.cif').sequences
    hit = tcs.find_fused_peptide(seqs).iloc[0]
    assert hit['peptide'] == 'GQVELGGGNAVEVCKGS' and hit['evidence'].startswith('sequence')


def test_split_fused_peptide():
    raw = TCRFrame.read(DATA / '6XCP' / 'rcsb.cif')
    split = tcs.split_fused_peptide(raw)
    assert split.chains == ['A', 'B', 'C', 'D', 'E']
    assert split.sequences['C'] == 'GQVELGGGNAVEVCKGS'
    assert split.sequences['B'] == raw.sequences['B'][17:]
    assert tcs.verify(split).attrs['valid']
    assert split.mhc_class == 'II' and len(split.peptide_core_sequence) == 9
    with pytest.warns(tcs.TCRStructWarning, match='no fused peptide'):
        tcs.split_fused_peptide(DATA / '1AO7' / 'rcsb.pdb')


def test_cli_verify_and_split(tmp_path, capsys):
    assert main(['verify', str(DATA / '6XCP' / 'rcsb.cif')]) == 0
    assert 'NOT valid' in capsys.readouterr().out
    out = tmp_path / '6xcp_split.cif'
    assert main(['split-peptide', str(DATA / '6XCP' / 'rcsb.cif'), str(out)]) == 0
    assert main(['verify', str(out)]) == 0
    assert capsys.readouterr().out.rstrip().endswith('valid')


# -------------------------------------------------------------------- sources

@needs_network
def test_tcr3d_tables():
    cx = tcs.sources.tcr3d_complexes('II')
    assert set(cx['mhc_class']) == {'II'} and '6XCP' in set(cx['pdb_id'])
    assert cx.set_index('pdb_id').at['6XCP', 'Epitope'] == 'GQVELGGGNAVEVCKGS'
    pm = tcs.sources.tcr3d_pmhc('I')
    assert set(pm['mhc_class']) == {'I'} and len(pm) > 1000


@needs_network
def test_rcsb_info_and_download(tmp_path):
    info = tcs.sources.rcsb_info('1ao7')
    assert info.at[0, 'pdb_id'] == '1AO7' and info.at[0, 'resolution'] == pytest.approx(2.6)
    with pytest.warns(tcs.TCRStructWarning, match='failed'):
        got = tcs.sources.rcsb_download(['1ao7', 'zzzz'], tmp_path, n_jobs=2)
    assert list(got['status'].str.split(':').str[0]) == ['downloaded', 'failed']
    assert Path(got.at[0, 'file']).name == '1AO7-assembly1.cif'        # assembly 1 by default
    assert TCRFrame.read(got.at[0, 'file']).peptide == 'LLFGYPVYV'
    deposited = tcs.sources.rcsb_download('1AO7', tmp_path, assembly=None)
    assert Path(deposited.at[0, 'file']).name == '1AO7.cif'
    fasta = tcs.sources.rcsb_download('1AO7', tmp_path, fmt='fasta')
    assert Path(fasta.at[0, 'file']).name == '1AO7.fasta'
    again = tcs.sources.rcsb_download('1AO7', tmp_path)
    assert again.at[0, 'status'] == 'exists'
    s = TCRFrame.from_rcsb('1ao7', metadata=True)
    assert s.metadata['resolution'] == pytest.approx(2.6) and s.chains == ['A', 'C', 'D', 'E', 'F']
    assert s.metadata['assembly'] == 1
    assert TCRFrame.from_rcsb('1ao7', assembly=None).metadata['assembly'] is None
    assert s.select(regions='cdr3').metadata == s.metadata


@needs_network
def test_from_rcsb_split_and_verify():
    pf = TCRFrame.from_rcsb('6XCP')
    assert 'imgt' in pf.df and pf.name == '6XCP'
    assert tcs.verify(tcs.split_fused_peptide(pf)).attrs['valid']


def test_tcr_methods():
    pf = TCRFrame.read(DATA / '6XCP' / 'rcsb.cif')
    assert not pf.is_valid()
    assert pf.verify().equals(tcs.verify(pf))
    assert pf.find_fused_peptide().equals(tcs.find_fused_peptide(pf))
    split = pf.split_fused_peptide()
    assert split.peptide == 'GQVELGGGNAVEVCKGS' and split.is_valid()
    assert TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').verify(kind='pmhc').attrs['valid']
    assert pf.verify(as_bool=True) is False and split.verify(as_bool=True) is True
    assert tcs.verify(DATA / '1AO7' / 'rcsb.pdb', as_bool=True) is True
