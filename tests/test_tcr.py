"""TCRFrame: reading, contents, regions, views and warnings."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from pdbframe import PDBFrame

import tcrstruct as tcs
from tcrstruct import TCRFrame

DATA = Path(__file__).parent / 'data'

@pytest.fixture(scope='module')
def class_i():
    return TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb', name='1AO7')


@pytest.fixture(scope='module')
def class_ii():
    return TCRFrame.read(DATA / '3QIB' / 'rcsb.pdb', name='3QIB')


# ------------------------------------------------------------------- reading

def test_read_is_an_annotated_pdbframe(class_i):
    assert isinstance(class_i, TCRFrame) and isinstance(class_i, PDBFrame)
    assert class_i.chains == ['A', 'C', 'D', 'E', 'F']
    assert {'imgt', 'original_chain'} <= set(class_i.df.columns)
    # hydrogens are kept on reading; only the metrics drop them
    assert len(class_i) == len(PDBFrame.read(DATA / '1AO7' / 'rcsb.pdb'))


def test_tcrframe_from_a_plain_pdbframe():
    pf = PDBFrame.read(DATA / '7T2D' / 'boltz-2.pdb')
    t = TCRFrame(pf)
    assert pf.chains == ['A', 'B', 'C', 'D', 'E']   # pf itself is unchanged
    assert 'imgt' not in pf.df and 'imgt' in t.df
    assert t.chain_table.loc['D', 'original_chain'] == 'C'
    assert t.chain_table.loc['C', 'role'] == 'peptide'
    assert TCRFrame(t).df is t.df                    # already annotated: not redone
    assert tcs.annotate(pf).df.equals(t.df)


def test_derived_frames_are_tcrframes(class_i):
    derived = [class_i.select(chains='D'), class_i.copy(), class_i.tcr_up(),
               class_i.select(regions='cdr3'), class_i.regions['peptide_core'],
               class_i.select(chains='D') + class_i.select(chains='E')]
    assert all(type(d) is TCRFrame for d in derived)


def test_residues_have_imgt_and_region(class_i):
    res = class_i.residues
    plain = PDBFrame.residues.fget(class_i)
    assert list(res.columns) == [*plain.columns, 'imgt', 'original_chain', 'region']
    assert res[plain.columns].equals(plain)
    assert class_i.sequences == {c: ''.join(g['aa']) for c, g in plain.groupby('chain')}


# ------------------------------------------------------------------ contents

def test_class_i(class_i):
    t = class_i
    assert t.mhc_class == 'I'
    assert list(t.chain_table['role']) == ['MHC heavy chain', 'peptide', 'TCR alpha', 'TCR beta',
                                      'B2M']
    assert t.chain_table.loc['A', 'gene'] == 'A' and t.chain_table.loc['F', 'gene'] == 'B2M'
    assert t.peptide == t.peptide_core_sequence == 'LLFGYPVYV'
    # A6 TCR on HLA-A*02:01: ANARCI's germline genes, tcrkit's allele identity
    ch = t.chain_table
    assert ch.loc[['D', 'E'], 'gene'].tolist() == ['TRAV12-2', 'TRBV6-5']
    assert ch.loc[['D', 'E'], 'allele'].tolist() == ['TRAV12-2*01', 'TRBV6-5*01']
    assert ch.loc[['D', 'E'], 'j_gene'].tolist() == ['TRAJ24*02', 'TRBJ2-7*01']
    assert ch.loc['A', 'identity'] == 1.0
    assert ch.loc['D', 'identity'] == pytest.approx(0.9565, abs=1e-4)
    assert ch.loc['D', 'j_identity'] == 1.0
    assert ch.loc['C', ['identity', 'j_identity']].isna().all()
    assert ch.loc['A', ['j_gene']].tolist() == ['-'] and np.isnan(ch.loc['A', 'j_identity'])
    assert t.cdrs['CDRB3'] == 'ASRPGLAGGRPEQY'
    assert set(t.regions) == set(tcs.REGIONS)
    assert t.angles['crossing_angle'] == pytest.approx(33.53, abs=0.01)


def test_class_ii(class_ii):
    t = class_ii
    assert t.mhc_class == 'II'
    assert t.chain_table.loc['A', 'role'] == 'MHC alpha'
    assert (t.chain_table.loc['A', 'gene'], t.chain_table.loc['B', 'gene']) == ('EA', 'EB')   # mouse I-E k
    assert t.peptide == 'LIAYLKQATKG' and t.peptide_core_sequence == 'IAYLKQATK'
    assert t.cdrs == {'CDRA1': 'TTMRA', 'CDRA2': 'LASGT', 'CDRA3': 'AALRATGGNNKLT',
                      'CDRB1': 'KGHPV', 'CDRB2': 'FQNQEV', 'CDRB3': 'ASSLNWSQDTQY'}


def test_regions_and_residues(class_ii):
    t = class_ii
    sel = t.select(regions='cdr3')
    assert sel.chains == ['D', 'E']
    assert len(sel) == len(t.regions['CDRA3']) + len(t.regions['CDRB3'])
    res = t.residues
    assert set(res['region']) >= {'', 'peptide_core', 'peptide', 'CDRB3', 'helices', 'Va'}
    assert ''.join(res.query("region == 'CDRB3'")['aa']) == 'ASSLNWSQDTQY'
    with pytest.raises(KeyError, match='unknown region'):
        t.select(regions='CDR4')


def test_repr_and_summary(class_i):
    assert repr(class_i) == "TCRFrame '1AO7', MHC class I, chains A, C, D, E, F"
    assert 'masks' not in class_i.select(chains='C').__dict__   # nothing computed to show it
    text = class_i.report()
    assert 'MHC class I' in text and 'LLFGYPVYV' in text and 'CDRB3=ASRPGLAGGRPEQY' in text
    s = class_i.summary()
    assert s['mhc_class'] == 'I' and s['CDRA3'] == class_i.cdrs['CDRA3']


def test_tcr_compare_matches_compare(class_i):
    model = TCRFrame.read(DATA / '1AO7' / 'af3-tbm.cif')
    a = model.compare(class_i, name='1AO7')
    b = tcs.compare(DATA / '1AO7' / 'af3-tbm.cif', DATA / '1AO7' / 'rcsb.pdb', name='1AO7')
    values = tcs.COLUMNS[1:]
    assert a[values].astype(float).equals(b[values].astype(float))
    assert a[tcs.SETTINGS].equals(b[tcs.SETTINGS])


# -------------------------------------------------------------------- angles

# The Pierce lab's tcr_docking_angle (github.com/piercelab/tcr_docking_angle, commit
# 2159f9a) on these references, AHo-renumbered: crossing and incident angle.
PIERCE = {'1AO7': (33.5309, 10.9191), '6ZKW': (59.6459, 19.7249),
          '3QIB': (36.917, 11.7698), '7T2D': (54.4595, 12.5265)}
# tcrstruct, both methods: (crossing, incident)
ANGLES = {'1AO7': {'piercelab': (33.5309, 11.3070), 'v1': (33.5309, 11.6789)},
          '6ZKW': {'piercelab': (59.6459, 19.4152), 'v1': (59.6459, 11.0268)},
          '3QIB': {'piercelab': (36.9169, 10.4171), 'v1': (31.2525, 27.3134)},
          '7T2D': {'piercelab': (54.4594, 14.4692), 'v1': (56.8566, 22.1176)}}
# tcrstruct, 'piercelab' with one part taken from 'v1'
PARTS = {'3QIB': {'helices': (31.2525, 15.5900), 'axis': (36.9169, 17.5565)},
         '7T2D': {'helices': (56.8566, 13.6831), 'axis': (54.4594, 15.7201)}}


@pytest.fixture(scope='module')
def angle_refs():
    return {p: TCRFrame.read(DATA / p / 'rcsb.pdb', name=p) for p in ANGLES}


@pytest.mark.parametrize('pdb', ANGLES)
@pytest.mark.parametrize('method', tcs.angles.ANGLE_METHODS)
def test_angle_methods(angle_refs, pdb, method):
    a = angle_refs[pdb].docking_angles(method)
    want = ANGLES[pdb][method]
    assert (a['crossing_angle'], a['incident_angle']) == pytest.approx(want, abs=1e-3)
    if method == 'piercelab':
        assert angle_refs[pdb].angles == a      # the default


@pytest.mark.parametrize('pdb', ANGLES)
def test_pierce_method_matches_tcr_docking_angle(angle_refs, pdb):
    """Same helix residues, so the crossing angle is Pierce's; the incident angle
    differs only by FAST vs IMGT residue pairing for the rotation axis."""
    a = angle_refs[pdb].angles
    assert a['crossing_angle'] == pytest.approx(PIERCE[pdb][0], abs=1e-3)
    assert a['incident_angle'] == pytest.approx(PIERCE[pdb][1], abs=2.0)


@pytest.mark.parametrize('pdb', PARTS)
@pytest.mark.parametrize('part', ['helices', 'axis'])
def test_angle_parts_one_by_one(angle_refs, pdb, part):
    t = angle_refs[pdb]
    a = t.docking_angles(**{part: 'v1'})
    assert (a['crossing_angle'], a['incident_angle']) == pytest.approx(PARTS[pdb][part], abs=1e-3)
    assert t.docking_angles({part: 'v1'}) == a
    # 'v1' with that part from 'piercelab' instead
    mine = tcs.angles.ANGLE_PARTS[part]['piercelab']
    b = t.docking_angles('v1', **{part: mine})
    assert b == t.docking_angles({'method': 'v1', part: mine})
    assert b != t.docking_angles('v1')
    # every part set by hand is the method
    assert t.docking_angles('v1', helices='imgt', axis='piercelab', signed=True) == t.angles


def test_angle_settings():
    assert tcs.angles.angle_settings() == {'helices': 'imgt', 'axis': 'piercelab', 'signed': True}
    assert tcs.angles.angle_settings('v1', signed=True) == \
        {'helices': 'v1', 'axis': 'v1', 'signed': True}
    with pytest.raises(KeyError, match='unknown angle parts'):
        tcs.angles.angle_settings({'helix': 'piercelab'})
    with pytest.raises(ValueError, match='signed must be one of'):
        tcs.angles.angle_settings(signed='piercelab')
    with pytest.raises(ValueError, match='helices must be one of'):
        tcs.angles.angle_settings(helices='piercelab')


def test_pierce_helices_are_pierces_residues(class_ii):
    """Standard numbering: the residue numbers of :data:`PIERCE_HELICES`."""
    df = class_ii.df
    ca = df[tcs.regions.mhc_helices(class_ii, method='imgt') & (df['atom_name'] == 'CA').to_numpy()]
    got = {c: sorted(g['residue_number']) for c, g in ca.groupby('chain')}
    want = {c: [n for lo, hi in r for n in range(lo, hi + 1)]
            for c, r in tcs.regions.PIERCE_HELICES['II'].items()}
    assert got == want


def test_swapped_tcr_chains_flip_only_the_signed_crossing_angle(class_i):
    """A TCR docked the other way round: 'piercelab' tells it apart, 'v1' does not."""
    swapped = class_i.rename_chains({'D': 'E', 'E': 'D'})
    for method in tcs.angles.ANGLE_METHODS:
        a = tcs.angles.docking_angles(class_i, method)
        b = tcs.angles.docking_angles(swapped, method)
        assert b['incident_angle'] == pytest.approx(a['incident_angle'], abs=1e-6)
        want = 180 - a['crossing_angle'] if method == 'piercelab' else a['crossing_angle']
        assert b['crossing_angle'] == pytest.approx(want, abs=1e-6)


def test_signed_angles_do_not_depend_on_the_peptide(class_i):
    """The groove axis is oriented by the alpha1 helix, so a peptide modelled backwards (or
    missing) leaves the 'piercelab' angles unchanged."""
    no_peptide = class_i.select(chains=[c for c in class_i.chains if c != 'C'])
    a = tcs.angles.docking_angles(class_i)
    b = tcs.angles.docking_angles(no_peptide)
    assert b == pytest.approx(a, abs=1e-9)


def test_angle_method_must_be_known(class_i):
    with pytest.raises(ValueError, match='method must be one of'):
        class_i.docking_angles('fast')
    with pytest.raises(ValueError, match='angle_method must be one of'):
        tcs.compare(class_i, class_i, angle_method='fast')


# --------------------------------------------------------------------- views

def test_view_marks_regions(class_ii):
    v = class_ii.view()
    html = v._repr_html_() if hasattr(v, '_repr_html_') else str(v)
    for name in ['peptide_core', 'CDRA1', 'CDRB3', 'helices']:
        assert name in html
    only = tcs.view(class_ii, regions='cdr3', extra_regions={'site': class_ii.select(chains='C')})
    html = only._repr_html_() if hasattr(only, '_repr_html_') else str(only)
    assert 'CDRB3' in html and 'site' in html and 'CDRA1' not in html


def test_view_comparison(class_i):
    v = tcs.view_comparison(DATA / '1AO7' / 'boltz-2.pdb', class_i)
    assert v is not None


# ------------------------------------------------------------------ warnings

def test_unrelated_chains_warn():
    # a class I model against a class II reference: every chain is unrelated, so
    # each one is reported, then the pair cannot be scored at all
    with pytest.warns(tcs.TCRStructWarning, match='chain .* identical'), \
            pytest.raises(ValueError, match='no chains'):
        tcs.compare(DATA / '1AO7' / 'tcrmodel2.pdb', DATA / '3QIB' / 'rcsb.pdb')


def test_missing_region_warns(class_i):
    pep_only = PDBFrame(class_i.df[class_i.df['chain'].isin(['A', 'C', 'F'])], name='pmhc')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tcs.view(pep_only, regions='cdrs')
    assert any('no' in str(w.message) and 'CDRA1' in str(w.message) for w in caught)


# --------------------------------------------------------------- orientation

@pytest.mark.parametrize('pdb', ['1AO7', '3QIB', '7N2N'])
def test_tcr_up_orientation(pdb):
    o = TCRFrame.read(DATA / pdb / 'rcsb.pdb').tcr_up()
    centre = {c: o.select(chains=c).coords.mean(axis=0) for c in o.chains}
    assert centre['D'][1] > 20 and centre['E'][1] > 20      # TCR up
    assert centre['A'][1] < 0                               # MHC below
    pep = o.select(chains='C', atoms='ca').coords
    assert pep[-1, 0] - pep[0, 0] > 15                      # peptide N -> C left to right
    R, _ = tcs.angles.tcr_up_orientation(o)
    assert np.allclose(R, np.eye(3), atol=1e-6)             # already standard


def test_normalize_7n2n_insertions():
    norm = tcs.normalize(DATA / '7N2N' / 'rcsb.pdb')
    d = norm.residues.query("chain == 'D' and 110 <= residue_number <= 113")
    assert list(d['residue_number'].astype(str) + d['insertion_code']) == \
        ['110', '111', '111A', '112A', '112', '113']
    assert norm.cdrs['CDRA3'] == 'AVLSPVQETSGSRLT'


def test_one_tcr_chain_keeps_its_cdrs():
    alpha = TCRFrame.read(DATA / '7N2N' / 'rcsb.pdb', name='7N2N').select(chains='D')
    assert list(alpha.cdrs) == ['CDRA1', 'CDRA2', 'CDRA3']
    # a lone alpha chain finds them too, without the rest of the complex
    assert TCRFrame(PDBFrame._wrap(alpha.df, 'alpha')).cdrs == alpha.cdrs
    res = alpha.residues
    cdr3 = res.query("region == 'CDRA3'")
    assert ''.join(cdr3['aa']) == 'AVLSPVQETSGSRLT'
    assert '111A' in set(cdr3['imgt']) and '112A' in set(cdr3['imgt'])


def test_region_names_ignore_case(class_ii):
    t = class_ii
    assert t.select(regions='cdra1').sequences == {'D': 'TTMRA'}
    assert t.select(regions='tcra').chains == ['D']
    assert t.select(regions=['CDR3', 'peptide_core']).sequences == {'C': 'IAYLKQATK', 'D': 'AALRATGGNNKLT',
                                                     'E': 'ASSLNWSQDTQY'}
    assert t.select(regions='pmhc').chains == ['A', 'B', 'C']
    assert 'cdras' in t.describe().index
    assert t.select(chains='E', regions='cdrs').sequences == {'E': 'KGHPVFQNQEVASSLNWSQDTQY'}


def test_fragments_keep_the_regions_of_the_complex(class_ii):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        helices = class_ii.regions['helices'].residues
        core = class_ii.peptide_core.residues
    # a fragment alone has no MHC to find helices on, so it must not try
    assert not caught
    assert set(helices['region']) == {'helices'} and set(core['region']) == {'peptide_core'}


def test_view_comparison_without_superposition(class_i):
    assert tcs.view_comparison(DATA / '1AO7' / 'boltz-2.pdb', class_i, superpose=False)


def _html(v):
    return v._repr_html_() if hasattr(v, '_repr_html_') else str(v)


def test_view_comparison_several_models(class_i):
    v = tcs.view_comparison({'AF3': DATA / '1AO7' / 'af3-tbm.cif',
                             'Boltz-2': DATA / '1AO7' / 'boltz-2.pdb'}, class_i,
                            superpose=['A', 'C'])
    html = _html(v)
    for text in ['AF3', 'Boltz-2', 'TCR up', 'Groove from above', 'CDRs']:
        assert text in html


def test_view_buttons_and_hidden_regions(class_ii):
    html = _html(class_ii.view(regions='all', hidden_regions=['peptide_centric', 'Va', 'Vb']))
    assert 'TCR up' in html and 'peptide_centric' in html


def test_view_of_a_selection(class_i):
    """A selection marks the default regions it has, without a warning, and is
    oriented by its complex (it has no helices of its own)."""
    part = class_i.select(regions=['cdr3', 'peptide_core'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        html = _html(part.view(tcr_up=True))
    assert 'TCR up' in html and 'CDRB3' in html and 'CDRA1' not in html
    with pytest.warns(tcs.TCRStructWarning, match="no \\['helices'\\] to mark"):
        part.view(regions='helices')


def test_groove_needs_helices(class_i):
    with pytest.raises(ValueError, match='groove-helix'):
        tcs.angles.orientations(class_i.select(chains=['C', 'D', 'E']))


def test_orientations_are_rotations(class_i):
    for R, t in tcs.angles.orientations(class_i).values():
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)


def _data(v):
    return json.loads(re.search(r'var D = (\{.*?\});\n', v._repr_html_(), re.S).group(1))


def _arrows(v):
    return _data(v)['arrows']


def test_view_vectors(class_i):
    arrows = _arrows(class_i.view(vectors=True))
    assert [a['label'].split(':')[0] for a in arrows] == \
        ['groove axis', 'groove normal', 'TCR alpha->beta', 'rotation axis']
    a = class_i.angles
    assert f"crossing {a['crossing_angle']:.1f}°" in arrows[2]['label']
    assert f"incident {a['incident_angle']:.1f}°" in arrows[3]['label']
    # drawn in the file's coordinates: the normal starts at the groove centre
    v = tcs.angles.docking_vectors(class_i)
    normal = arrows[1]
    assert normal['start'] == pytest.approx(v['groove_center'].tolist(), abs=1e-6)
    assert np.subtract(normal['end'], normal['start']) == \
        pytest.approx((20 * v['groove_normal']).tolist(), abs=1e-6)
    folded = _arrows(class_i.view(vectors='v1'))
    assert folded[2]['label'].startswith('TCR beta->alpha')
    assert not _arrows(class_i.view())


def test_views_keep_the_file_coordinates(class_i):
    plain, up = _data(class_i.view()), _data(class_i.view(tcr_up=True))
    assert plain['models'] == up['models'] == _data(PDBFrame.view(class_i))['models']
    assert set(plain['orientations']) == {'TCR up', 'Groove from above'}
    assert plain['orientation'] is None
    R, t = tcs.angles.tcr_up_orientation(class_i)
    assert np.array(up['orientation'][0]) == pytest.approx(R)      # only the camera turns



def test_sequences_only_regions():
    """Without coordinates there is no peptide-centric shell, and no class II core
    (it is found by distance), with a warning instead of a wrong window."""
    seqs = TCRFrame.read(DATA / '3QIB' / 'rcsb.pdb').sequences
    with pytest.warns(tcs.TCRStructWarning, match='needs coordinates'):
        masks = TCRFrame.from_sequences(seqs, name='3QIB').masks
    assert 'peptide_core' not in masks and 'peptide_centric' not in masks
    seqs = TCRFrame.read(DATA / '1AO7' / 'rcsb.pdb').sequences
    t = TCRFrame.from_sequences(seqs, name='1AO7')
    assert 'peptide_centric' not in t.masks and t.peptide_core_sequence == 'LLFGYPVYV'


def test_list_regions():
    """One row per region, saying how it is defined; the CDR and helix positions are
    tcrkit's IMGT definitions."""
    import tcrkit
    table = tcs.list_regions()
    assert list(table.index) == tcs.REGIONS
    assert set(table['defined_by'].str.split(' |:').str[0]) == {'IMGT', 'whole', 'distance'}
    cdrs = tcrkit.imgt_regions('tcr').set_index('region')
    for name in tcs.regions.CDRS:
        lo, hi = cdrs.loc[f'CDR{name[-1]}', ['imgt_start', 'imgt_end']]
        assert table.loc[name, 'imgt'] == f'{lo}-{hi}'
    assert 'G-BETA 52-86 without 61B, 62' in table.loc['helices', 'imgt']
    groups = tcs.list_region_groups()
    assert list(groups.index) == list(tcs.REGION_GROUPS)
    assert groups.loc['cdr3', 'regions'] == 'CDRA3, CDRB3'


# ------------------------------------------------------- sequences, contacts

def test_sequence_of_chains_and_regions(class_i, class_ii):
    assert class_i.sequence('C') == class_i.peptide
    assert class_i.sequence('cdrb3') == class_i.cdrs['CDRB3'] == 'ASRPGLAGGRPEQY'
    assert class_ii.sequence('peptide_core') == class_ii.peptide_core_sequence == 'IAYLKQATK'
    assert class_i.sequence('B') is None                        # no MHC beta in class I
    assert class_i.summary()['TRAV'] == 'TRAV12-2*01'
    with pytest.raises(ValueError, match='spans chains'):
        class_i.sequence('peptide_centric')
    with pytest.raises(ValueError, match='group'):
        class_i.sequence('cdr3')
    with pytest.raises(KeyError, match='unknown region'):
        class_i.sequence('CDR4')


def test_contact_map(class_i, class_ii):
    table = class_i.contact_map()
    assert (table['distance'] <= 5.0).all()
    assert set(table['row_region']) <= set(tcs.regions.CDRS)
    assert ('CDRA1 Q37', 'L2') in set(zip(table['row'], table['column']))
    grid = class_i.contact_map(grid=True)
    assert list(grid.columns) == ['L1', 'L2', 'F3', 'G4', 'Y5', 'P6', 'V7', 'Y8', 'V9']
    assert grid.loc['CDRA1 Q37', 'L2'] == pytest.approx(
        table.query("row == 'CDRA1 Q37' and column == 'L2'")['distance'].min())
    assert len(grid) == table['row'].nunique()
    # class II: positions count from the core (IAYLKQATK), the flanks at -1 and 10
    cols = list(class_ii.contact_map(grid=True).columns)
    assert cols == ['L-1', 'I1', 'A2', 'Y3', 'L4', 'K5', 'Q6', 'A7', 'T8', 'K9', 'G10']
    tight = class_i.contact_map(cutoff=3.5)
    assert len(tight) < len(table)


def test_plot_contact_map(class_i):
    import matplotlib
    matplotlib.use('Agg')
    ax = tcs.plot_contact_map(class_i)
    grid = class_i.contact_map(grid=True)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(grid.columns)
    assert tcs.plot_contact_map(grid).get_title() == ''
