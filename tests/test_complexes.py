"""Several complexes in one file, splitting a complex into parts, and the structure checks."""

from pathlib import Path

import pytest

import tcrstruct as tcs
from tcrstruct import TCRFrame
from tcrstruct.cli import main

DATA = Path(__file__).parent / 'data'
MULTI = DATA / '2E7L' / 'rcsb.cif'


def _status(table) -> dict:
    return dict(zip(table['check'], table['status']))


# ------------------------------------------------------------------ complexes

def test_complex_table_groups_chains_by_contacts():
    table = tcs.complex_table(MULTI)
    assert table[['mhc', 'mhc_partner', 'peptide', 'tcr_alpha', 'tcr_beta']].values.tolist() == \
        [['E', '-', 'Q', 'A', 'D'], ['F', '-', 'P', 'B', 'C']]
    assert list(table.index) == [1, 2] and table.attrs['unused'] == []


def test_find_complexes():
    found = tcs.find_complexes(MULTI)
    assert [f.name for f in found] == ['rcsb_1', 'rcsb_2']
    for frame, chains in zip(found, [['E', 'Q', 'A', 'D'], ['F', 'P', 'B', 'C']]):
        assert frame.chains == ['A', 'C', 'D', 'E']
        assert frame.metadata['complex_chains'] == chains
        status = _status(frame.verify())
        assert status['peptide in the MHC groove'] == status['TCR alpha-beta pairing'] == 'ok'
    # the same from read(complex=)
    second = TCRFrame.read(MULTI, complex=2)
    assert second.metadata['complex_chains'] == ['F', 'P', 'B', 'C']
    with pytest.raises(ValueError, match='it has 2 complexes'):
        TCRFrame.read(MULTI, complex=3)


def test_read_keeps_one_complex():
    with pytest.warns(tcs.TCRStructWarning, match='2 copies of the complex .* kept complex 1 '
                      r'\(file chains E, Q, A, D\) and left out chains B, C, F, P'):
        first = TCRFrame.read(MULTI)
    assert first.chains == ['A', 'C', 'D', 'E']
    assert first.metadata['complex'] == 1 and first.metadata['n_complexes'] == 2
    assert dict(zip(first.chain_table.index, first.chain_table['original_chain'])) == \
        {'A': 'E', 'C': 'Q', 'D': 'A', 'E': 'D'}
    # complex=None: every chain, the first of each kind by chain ID, mixing the copies
    every = TCRFrame.read(MULTI, complex=None)
    assert every.chains == ['A', 'C', 'D', 'E', 'G', 'H', 'I', 'J']
    assert dict(zip(every.chain_table.index[:4], every.chain_table['original_chain'][:4])) == \
        {'A': 'E', 'C': 'P', 'D': 'A', 'E': 'C'}
    with pytest.raises(ValueError, match='numbered from 1'):
        TCRFrame.read(MULTI, complex=0)
    with pytest.raises(ValueError, match='it has one complex'):
        TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb', complex=2)


def test_find_complexes_of_an_annotated_frame():
    annotated = TCRFrame.read(MULTI, complex=None)        # A-F mixes the two copies
    assert [f.metadata['complex_chains'] for f in tcs.find_complexes(annotated)] == \
        [['E', 'Q', 'A', 'D'], ['F', 'P', 'B', 'C']]


def test_one_complex_files():
    one = tcs.find_complexes(DATA / '1AO7' / 'rcsb.pdb')
    assert len(one) == 1 and one[0].chains == TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').chains


def test_unused_chains_warn():
    tcr_only = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').select(chains=['A', 'C', 'D', 'E', 'F'])
    far = tcr_only.df.copy()
    tcr = far['chain'].isin(['D', 'E'])
    far.loc[tcr, 'x'] = far.loc[tcr, 'x'] + 200.0
    with pytest.warns(tcs.TCRStructWarning, match='chains in no complex: D .*, E '):
        found = tcs.find_complexes(TCRFrame(tcr_only._same(far)))
    assert found[0].chains == ['A', 'C', 'F']


def test_cli_complexes(tmp_path, capsys):
    main(['-q', 'complexes', str(MULTI)])
    assert 'F           -       P         B        C' in capsys.readouterr().out
    main(['-q', 'complexes', str(MULTI), '-o', str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ['rcsb_1.cif', 'rcsb_2.cif']
    assert TCRFrame.read(tmp_path / 'rcsb_2.cif').peptide == TCRFrame.read(MULTI, complex=2).peptide


# ---------------------------------------------------------------------- split

def test_split():
    s = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb', name='1AO7')
    parts = s.split()
    assert {k: v.chains for k, v in parts.items()} == {
        'tcr': ['D', 'E'], 'pmhc': ['A', 'C', 'F'], 'mhc': ['A', 'F'], 'peptide': ['C']}
    assert len(parts['tcr']) + len(parts['pmhc']) == len(s)
    cdrs = tcs.split(s, parts='cdrs')['cdrs']
    assert set(cdrs.residues['region']) == set(tcs.regions.CDRS)


def test_split_writes_files(tmp_path):
    s = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb', name='1AO7')
    paths = s.split(['tcr', 'peptide_core'], out_dir=tmp_path, fmt='cif')
    assert [p.name for p in paths.values()] == ['1AO7_tcr.cif', '1AO7_peptide_core.cif']
    assert TCRFrame.read(paths['peptide_core']).peptide == 'LLFGYPVYV'


def test_split_missing_part_warns():
    pmhc = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').select(chains=['A', 'C', 'F'])
    with pytest.warns(tcs.TCRStructWarning, match='no tcr'):
        parts = pmhc.split()
    assert set(parts) == {'pmhc', 'mhc', 'peptide'}


def test_cli_split(tmp_path, capsys):
    main(['-q', 'split', str(MULTI), str(tmp_path), '--complex', '2', '--parts', 'tcr', 'pmhc'])
    assert sorted(p.name for p in tmp_path.iterdir()) == ['rcsb_pmhc.cif', 'rcsb_tcr.cif']
    second = TCRFrame.read(MULTI, complex=2)
    tcr = TCRFrame.read(tmp_path / 'rcsb_tcr.cif')
    assert tcr.sequences == {c: second.sequences[c] for c in ('D', 'E')}


# ----------------------------------------------------------- structure checks

def test_verify_copies():
    v = tcs.verify(MULTI, complex=2).set_index('check')
    assert v.at['copies', 'status'] == 'warning'
    assert v.at['copies', 'detail'].startswith('complex 2 of 2 in the file (by contacts), '
                                               'file chains F, P, B, C')
    assert v.at['peptide in the MHC groove', 'status'] == 'ok'
    assert 'extra chains' not in v.index and 'one complex' not in v.index
    # every chain kept: A-F mix the copies
    mixed = tcs.verify(MULTI, complex=None).set_index('check')
    assert mixed.at['one complex', 'status'] == 'fail'
    assert 'complex=n' in mixed.at['one complex', 'detail']
    ok = tcs.verify(DATA / '1AO7' / 'rcsb.pdb')
    assert not {'copies', 'one complex'} & set(ok['check'])


def test_chain_breaks():
    s = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb', name='1AO7')
    assert _status(tcs.verify(DATA / '7T2D' / 'af3-tbm.cif'))['chain breaks'] == 'ok'
    # take out 3 residues of CDR3 beta
    cut = s.residues.query("chain == 'E' and region == 'CDRB3'")['observed_index'].iloc[4:7]
    gapped = s.select(mask=~(s.df['chain'].eq('E') & s.df['observed_index'].isin(cut)).to_numpy())
    detail = tcs.verify(gapped).set_index('check').at['chain breaks', 'detail']
    assert 'E 108-112A (CDRB3)' in detail and 'at least 1 missing' in detail


def test_tcr_anchors():
    s = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb')
    assert _status(tcs.verify(s))['TCR anchors'] == 'ok'
    no_cys = s.select(mask=~(s.df['chain'].eq('D') & s.df['imgt'].eq('104')).to_numpy())
    v = tcs.verify(no_cys).set_index('check')
    assert v.at['TCR anchors', 'status'] == 'warning'
    assert 'D 104: missing' in v.at['TCR anchors', 'detail']
    assert tcs.is_valid(no_cys)                     # structure checks only warn


@pytest.mark.parametrize('pdb, chain, number', [('1AO7', 'A', 164), ('3QIB', 'B', 79)])
def test_mhc_anchors(pdb, chain, number):
    s = TCRFrame.read(DATA / pdb / 'rcsb.pdb', name=pdb)
    v = tcs.verify(s).set_index('check')
    assert v.at['MHC anchors', 'status'] == 'ok'
    no_cys = s.select(mask=~(s.df['chain'].eq(chain) & s.df['residue_number'].eq(number)).to_numpy())
    v = tcs.verify(no_cys).set_index('check')
    assert v.at['MHC anchors', 'status'] == 'warning'
    assert f'{chain} ' in v.at['MHC anchors', 'detail'] and '74: missing' in v.at['MHC anchors', 'detail']


def test_tcr_numbering(tmp_path):
    raw = DATA / '7T2D' / 'af3-tbm.cif'
    assert 'not IMGT numbered' in tcs.verify(raw).set_index('check').at['TCR numbering', 'detail']
    normalized = tcs.normalize(raw)
    path = normalized.write(tmp_path / 'n.pdb')
    v = tcs.verify(path).set_index('check')
    assert v.at['TCR numbering', 'status'] == 'ok' and 'matches' in v.at['TCR numbering', 'detail']
    # shift a few residue numbers: they no longer match the recomputed IMGT labels
    df = normalized.df.copy()
    shift = df['chain'].eq('D') & df['residue_number'].between(1, 5)
    df.loc[shift, 'residue_number'] += 1000
    bad = TCRFrame(normalized._same(df))
    v = tcs.verify(bad).set_index('check')
    assert v.at['TCR numbering', 'status'] == 'warning'
    assert 'differ' in v.at['TCR numbering', 'detail']


def test_structure_checks_can_be_skipped():
    v = tcs.verify(DATA / '1AO7' / 'rcsb.pdb', structure_checks=False)
    assert not {'chain breaks', 'MHC anchors', 'TCR anchors', 'TCR numbering'} & set(v['check'])


# ---------------------------------------------------------------- TCR docking

def test_cdr_contacts():
    table = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').cdr_contacts()
    assert list(table.index) == ['CDRA1', 'CDRA2', 'CDRA3', 'CDRB1', 'CDRB2', 'CDRB3']
    assert table.loc['CDRA3', 'peptide_distance'] < 4.0 and table['peptide_residues'].sum() == 14
    assert table['touches_pmhc'].sum() == 5


def test_docking_check():
    ok = tcs.verify(DATA / '1AO7' / 'rcsb.pdb').set_index('check')
    assert ok.at['TCR docking', 'status'] == 'ok'
    with pytest.warns(tcs.TCRStructWarning, match='2 copies'):
        bad = tcs.verify(DATA / '9RUP' / 'rcsb.cif').set_index('check')
    assert bad.at['TCR on the pMHC', 'status'] == 'ok'          # it does touch, barely
    assert bad.at['TCR docking', 'status'] == 'warning'
    assert bad.at['TCR docking', 'detail'].startswith(
        'only 1 CDR residue on the peptide; only 3 of 6 CDRs on the pMHC')
    assert bad.attrs['valid']                                   # a warning, not a failure
