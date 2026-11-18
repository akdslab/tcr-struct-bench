"""Checking that a structure (or a set of sequences) is a complete TCR-pMHC or pMHC.

    >>> tcs.verify('6xcp.cif')                       # one row per check
    >>> tcs.verify(pf, kind='pmhc')                  # no TCR expected
    >>> tcs.verify(pf, as_bool=True)                 # just True / False
    >>> tcs.verify(seqs)                             # sequences only: chain checks
    >>> tcs.verify_dir('structures/', n_jobs=8)      # one row per file

Checks, each ``'ok'``, ``'warning'`` or ``'fail'``:

- **MHC class:** an MHC chain is found and its class is known.
- **Chains:** every chain the class needs is present.
  - class I: heavy chain A, peptide C, B2M F;
  - class II: alpha A, beta B, peptide C;
  - TCR-pMHC also needs TCR alpha D and beta E.

  A missing B2M is a warning (a failure with ``require_b2m=True``).
- **Peptide:** found and of a usual length (class I 8-15, class II 9-30). A missing
  peptide that is fused to another chain is reported with a pointer to
  :func:`.fused.split_fused_peptide`.
- **Copies**, for a file with several copies of the complex: which one was checked
  (complex 1 by default, ``complex=n`` for another; :func:`.complexes.complex_table`)
  and the chains left out - a warning.
- **Extra chains:** chains kept beside A-F (other molecules) - a warning.
- **One complex**, with ``complex=None`` (every chain kept): the chains picked as A-F
  are one complex by contacts; a fail if they mix copies.
- **Contacts**, for structures: residue pairs within `cutoff` Å (heavy atoms):
  - peptide in the MHC groove (at least half its residues touch the MHC);
  - TCR on the pMHC (CDRs touching both the peptide and the MHC);
  - TCR alpha paired with beta;
  - B2M with the class I heavy chain;
  - class II alpha with beta;
  - TCR docking (a warning): fewer than 4 of the 6 CDRs touching the pMHC, fewer than 3
    CDR residues on the peptide, or a crossing angle over 90 degrees (reverse docking).
    In a set of 320 RCSB TCR-pMHC structures this flags only 9RUP (a TCR barely
    on its pMHC) and six reverse-docked TCRs (4Y19, 4Y1A, 5SWS, 5SWZ, 7JWI, 9GV7).
- **Structure**, for structures (``structure_checks=True``), warnings only:
  - chain breaks: consecutive Cα atoms more than 4.2 Å apart in chains A-F, with
    the fewest residues that can be missing (the gap over 3.8 Å per residue) and the
    region they fall in;
  - MHC anchors: the disulfide cysteines of the second groove domain, Cys 11 and
    Cys 74 of G-ALPHA2 (class I) or G-BETA (class II), at their IMGT G-domain positions;
  - TCR anchors: the conserved IMGT residues of each V domain, Cys 23, Trp 41,
    Cys 104 and Phe / Trp 118 (Cys 23 and 104 orient the docking angles);
  - TCR numbering: whether the file's residue numbers are IMGT; if they are (a
    normalized file), any residue whose number differs from the recomputed IMGT
    label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure.geometry import atom_pairs_within

from .chains import annotate, detect_chains, heavy_atoms

__all__ = ['verify', 'verify_dir', 'is_valid']

_PEPTIDE_LENGTH = {'I': (8, 15), 'II': (9, 30)}
_ROLE = {'A': 'MHC heavy chain / alpha', 'B': 'MHC beta', 'C': 'peptide', 'D': 'TCR alpha',
         'E': 'TCR beta', 'F': 'B2M'}


def _observed(pf) -> pd.DataFrame:
    """``pf.residues`` (a TCRFrame's, with ``imgt`` and ``region``) of the residues with atoms."""
    res = pf.residues
    return res[res['observed']].astype({'observed_index': int}).reset_index(drop=True)


def _row(check, status, detail):
    return {'check': check, 'status': status, 'detail': detail}


def _required(mhc_class: str, kind: str) -> list[str]:
    need = ['A', 'C', 'F'] if mhc_class == 'I' else ['A', 'B', 'C']
    return need + (['D', 'E'] if kind == 'tcrpmhc' else [])


def _residue_contacts(pf: PDBFrame, a, b, cutoff: float) -> tuple[int, int, int]:
    """Residues of `a` touching `b`, residues of `b` touching `a`, residue pairs."""
    df = pf.df
    ma, mb = np.asarray(a, bool), np.asarray(b, bool)
    if not ma.any() or not mb.any():
        return 0, 0, 0
    i, j, _ = atom_pairs_within(pf.coords[ma], pf.coords[mb], cutoff)
    ra = (df['chain'] + ':' + df['observed_index'].astype(str)).to_numpy()
    pa, pb = ra[ma][i], ra[mb][j]
    return len(set(pa)), len(set(pb)), len(set(zip(pa, pb)))


def _contacts(pf: PDBFrame, kind: str, cutoff: float) -> list[dict]:
    original = pf
    pf = heavy_atoms(pf)
    df, rows = pf.df, []

    def chain(*ids):
        return df['chain'].isin(ids).to_numpy()

    chains = set(pf.chains)
    if 'C' in chains and chains & {'A', 'B'}:
        n_pep = df.loc[chain('C'), 'observed_index'].nunique()
        touching, _, _ = _residue_contacts(pf, chain('C'), chain('A', 'B'), cutoff)
        ok = touching >= n_pep / 2
        rows.append(_row('peptide in the MHC groove', 'ok' if ok else 'fail',
                         f'{touching} of {n_pep} peptide residues touch the MHC'))
    if kind == 'tcrpmhc' and {'D', 'E'} <= chains:
        cdr = np.logical_or.reduce([pf.masks.get(c, np.zeros(len(df), bool))
                                    for c in ('CDRA1', 'CDRA2', 'CDRA3', 'CDRB1', 'CDRB2',
                                              'CDRB3')])
        on_pep, _, _ = _residue_contacts(pf, cdr, chain('C'), cutoff)
        on_mhc, _, _ = _residue_contacts(pf, cdr, chain('A', 'B'), cutoff)
        ok = on_pep > 0 and on_mhc > 0
        rows.append(_row('TCR on the pMHC', 'ok' if ok else 'fail',
                         f'{on_pep} CDR residues touch the peptide, {on_mhc} touch the MHC'))
        if ok:
            rows.append(_docking(original, cutoff))
        _, _, pairs = _residue_contacts(pf, chain('D'), chain('E'), cutoff)
        rows.append(_row('TCR alpha-beta pairing', 'ok' if pairs >= 10 else 'fail',
                         f'{pairs} residue pairs in contact'))
    if {'A', 'F'} <= chains and 'B' not in chains:
        _, _, pairs = _residue_contacts(pf, chain('A'), chain('F'), cutoff)
        rows.append(_row('B2M on the heavy chain', 'ok' if pairs >= 10 else 'fail',
                         f'{pairs} residue pairs in contact'))
    if {'A', 'B'} <= chains:
        _, _, pairs = _residue_contacts(pf, chain('A'), chain('B'), cutoff)
        rows.append(_row('MHC alpha-beta pairing', 'ok' if pairs >= 10 else 'fail',
                         f'{pairs} residue pairs in contact'))
    return rows


#: TCR docking warnings: fewer CDR residues on the peptide, fewer CDRs touching the pMHC,
#: or a crossing angle above this ('piercelab', 0-180: a reverse-docked TCR).
DOCKING_MIN_PEPTIDE_RESIDUES = 3
DOCKING_MIN_CDRS = 4
DOCKING_MAX_CROSSING = 90.0


def _docking(pf, cutoff: float) -> dict:
    """Whether the TCR sits on the pMHC as TCRs usually do (warnings only)."""
    from .contacts import cdr_contacts
    table = cdr_contacts(pf, cutoff)
    touching = list(table.index[table['touches_pmhc']])
    on_pep = int(table['peptide_residues'].sum())
    angles = pf.angles                     # pf: the annotated TCRFrame
    crossing, incident = angles['crossing_angle'], angles['incident_angle']
    odd = []
    if on_pep < DOCKING_MIN_PEPTIDE_RESIDUES:
        odd.append(f'only {on_pep} CDR residue{"s" if on_pep != 1 else ""} on the peptide')
    if len(touching) < DOCKING_MIN_CDRS:
        odd.append(f'only {len(touching)} of 6 CDRs on the pMHC')
    if crossing > DOCKING_MAX_CROSSING:
        odd.append(f'crossing angle {crossing:.0f} degrees (reverse docking)')
    d3 = ', '.join(f'{c} {table.at[c, "peptide_distance"]:.1f} A' for c in ('CDRA3', 'CDRB3'))
    geometry = (f'CDR3s from the peptide: {d3}; crossing {crossing:.0f}, incident '
                f'{incident:.0f} degrees')
    if not odd:
        return _row('TCR docking', 'ok', f'{len(touching)} of 6 CDRs on the pMHC, {on_pep} CDR '
                    f'residues on the peptide; {geometry}')
    reverse_only = crossing > DOCKING_MAX_CROSSING and on_pep >= DOCKING_MIN_PEPTIDE_RESIDUES
    advice = ('reverse docking is rare but real (e.g. 5SWS, 4Y19)' if reverse_only
              else 'usually 5-6 CDRs touch the pMHC and about 10 CDR residues the peptide, '
              'so check the TCR is bound as in a real complex')
    return _row('TCR docking', 'warning', '; '.join(odd) + f' (CDRs on the pMHC: '
                f'{", ".join(touching) or "none"}; {geometry}); {advice}; tcs.cdr_contacts() '
                'gives each CDR')


#: Consecutive Cα atoms farther apart than this (Å) are a chain break.
BREAK_DISTANCE = 4.2

#: The conserved residues of a TCR V domain: IMGT position -> allowed amino acids.
TCR_ANCHORS = {'23': 'C', '41': 'W', '104': 'C', '118': 'FW'}

#: The groove-domain disulfide of the MHC: class -> (chain, G domain, IMGT positions).
MHC_ANCHORS = {'I': ('A', 'G-ALPHA2', ('11', '74')), 'II': ('B', 'G-BETA', ('11', '74'))}


def _one_complex(pf) -> dict:
    """Whether the chains annotated A-F are one complex of the file (by contacts)."""
    from .complexes import complex_table
    table = complex_table(pf)
    picked = {c: pf.df.loc[pf.df['chain'] == c, 'original_chain'].iloc[0]
              for c in 'ABCDEF' if c in pf.chains}
    original = set(picked.values())
    for number, row in table.iterrows():
        if original <= set(row['chains']):
            return _row('one complex', 'ok', f'chains A-F are complex {number} of {len(table)} '
                        '(by contacts); tcs.find_complexes() gives each')
    where = {c: next((str(n) for n, r in table.iterrows() if o in r['chains']), 'none')
             for c, o in picked.items()}
    mixed = ', '.join(f'{c} (file {picked[c]}): complex {where[c]}' for c in picked)
    return _row('one complex', 'fail', f'the chains picked mix copies: {mixed}; read one '
                f'with complex=n, n = 1-{len(table)}')


def _breaks(pf) -> dict:
    """Chain breaks in chains A-F: consecutive Cα atoms over :data:`BREAK_DISTANCE` apart."""
    from .tcr import TCRFrame
    df = pf.df
    ca = df[(df['atom_name'] == 'CA') & df['chain'].isin(list('ABCDEF'))]
    region = _observed(TCRFrame.ensure(pf)).set_index(['chain', 'observed_index'])['region']
    found = []
    for chain, g in ca.groupby('chain', sort=True):
        g = g.drop_duplicates('observed_index').sort_values('observed_index')
        xyz = g[['x', 'y', 'z']].to_numpy(float)
        if len(xyz) < 2:
            continue
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        for i in np.flatnonzero(d > BREAK_DISTANCE):
            a, b = g.iloc[i], g.iloc[i + 1]
            missing = max(int(np.ceil(d[i] / 3.8)) - 1, 1)     # a loop may hold more
            label = (f'{chain} {a["residue_number"]}{a["insertion_code"].strip()}-'
                     f'{b["residue_number"]}{b["insertion_code"].strip()}')
            where = {region.get((chain, a['observed_index'])),
                     region.get((chain, b['observed_index']))}
            where = sorted(w for w in where if isinstance(w, str) and w and w not in
                           ('TCRa', 'TCRb', 'MHC', 'B2M', 'peptide', '-'))
            if where:
                label += f' ({"/".join(where)})'
            found.append(f'{label}: {d[i]:.1f} A, at least {missing} missing')
    if not found:
        return _row('chain breaks', 'ok', 'no breaks in chains A-F')
    return _row('chain breaks', 'warning', f'{len(found)}: ' + '; '.join(found)
                + ' (residues missing there are not scored)')


def _anchors(pf) -> dict:
    """The conserved V-domain residues (:data:`TCR_ANCHORS`) of chains D and E."""
    res = _observed(pf)
    problems = []
    for chain in ('D', 'E'):
        r = res[res['chain'] == chain].set_index('imgt')['aa']
        for pos, allowed in TCR_ANCHORS.items():
            aa = r.get(pos)
            if aa is None:
                problems.append(f'{chain} {pos}: missing')
            elif aa not in allowed:
                problems.append(f'{chain} {pos}: {aa}, not {"/".join(allowed)}')
    if not problems:
        return _row('TCR anchors', 'ok', 'Cys23, Trp41, Cys104, Phe/Trp118 in both V domains')
    return _row('TCR anchors', 'warning', '; '.join(problems) + ' (a missing Cys23 or Cys104 '
                'leaves the docking angles NaN; check the V-domain numbering)')


def _mhc_anchors(pf) -> dict:
    """The disulfide cysteines of the MHC's second groove domain (:data:`MHC_ANCHORS`)."""
    from .mhc import mhc_numbering
    chain, domain, positions = MHC_ANCHORS[pf.mhc_class]
    num = mhc_numbering(pf)
    num = num[(num['chain'] == chain) & (num['domain'] == domain)]
    seen = {(r, i.strip()) for r, i in _observed(pf).query('chain == @chain')[
        ['residue_number', 'insertion_code']].itertuples(index=False)}
    num = num[[(r, i.strip()) in seen for r, i in
               zip(num['residue_number'], num['insertion_code'].fillna(''))]]
    aa = num.drop_duplicates('imgt').set_index('imgt')['aa']
    problems = [f'{chain} {domain} {pos}: ' + ('missing' if aa.get(pos) is None else
                                                f'{aa.get(pos)}, not C')
                for pos in positions if aa.get(pos) != 'C']
    what = f'Cys{positions[0]}, Cys{positions[1]} of {domain} (chain {chain})'
    if not problems:
        return _row('MHC anchors', 'ok', f'{what}, the groove disulfide')
    return _row('MHC anchors', 'warning', '; '.join(problems) + f' (expected {what}; '
                'check the MHC chain, or its G-domain numbering with tcs.mhc_numbering)')


def _numbering(pf) -> dict:
    """Whether the TCR chains are numbered by IMGT in the file, and if so correctly."""
    res = _observed(pf)
    v = res[res['chain'].isin(['D', 'E']) & (res['imgt'] != '-')
            & ~res['imgt'].str.startswith('-')]
    if v.empty:
        return _row('TCR numbering', 'ok', 'no IMGT-numbered TCR residues')
    file_label = v['residue_number'].astype(str) + v['insertion_code'].str.strip()
    same = (file_label == v['imgt']).to_numpy()
    if same.mean() < 0.5:
        return _row('TCR numbering', 'ok', 'the file is not IMGT numbered (labels are in the '
                    "imgt column; tcs.normalize() writes them as residue numbers)")
    if same.all():
        return _row('TCR numbering', 'ok', 'the file is IMGT numbered, and matches the '
                    'recomputed numbering')
    bad = v[~same]
    shown = ', '.join(f'{c} {n}{i.strip()} (IMGT {m})' for c, n, i, m in
                      bad[['chain', 'residue_number', 'insertion_code', 'imgt']].head(5)
                      .itertuples(index=False))
    return _row('TCR numbering', 'warning', f'the file is IMGT numbered, but {len(bad)} '
                f'residues differ from the recomputed numbering: {shown}'
                + (' ...' if len(bad) > 5 else '') + '; renumber with tcs.normalize()')


def verify(structure, kind: str = 'tcrpmhc', require_b2m: bool = False,
           contacts: bool = True, cutoff: float = 4.5, as_bool: bool = False,
           structure_checks: bool = True, complex: int | None = 1) -> pd.DataFrame | bool:
    """Check that a structure, or a set of sequences, is a complete TCR-pMHC (or pMHC).

    Args:
        structure: a file or PDBFrame, or sequences ``{chain: sequence}`` (then only
            the chain checks run).
        kind: ``'tcrpmhc'`` (TCR alpha and beta required) or ``'pmhc'``.
        require_b2m: for class I, a missing B2M is a failure (by default a warning).
        contacts: check the contacts between the parts (structures only).
        cutoff: contact distance in Å, between heavy atoms.
        as_bool: return only True (nothing failed) or False.
        structure_checks: check chain breaks, the TCR anchor residues and the TCR
            numbering (structures only; warnings, never failures).
        complex: for a file with several copies of the complex, which one to check
            (from 1); ``None`` checks the file as it is, every chain kept (see
            :func:`.chains.annotate`).

    Returns:
        One row per check: ``check``, ``status`` (``'ok'``, ``'warning'``, ``'fail'``)
        and ``detail``. ``result.attrs['valid']`` is True when nothing failed. With
        ``as_bool=True``, just that True / False.
    """
    from .fused import find_fused_peptide
    if kind not in ('tcrpmhc', 'pmhc'):
        raise ValueError(f"kind must be 'tcrpmhc' or 'pmhc', not {kind!r}")
    is_seq = isinstance(structure, dict)
    if is_seq:
        table = detect_chains(structure)
        pf, present = None, set(table['new_chain'])
        name = 'sequences'
        genes = dict(zip(table['new_chain'], table['details']))
        lengths = dict(zip(table['new_chain'], table['sequence'].str.len()))
    else:
        pf = annotate(structure, complex=complex)
        present, name = set(pf.chains), pf.name
        genes = dict(zip(pf.chain_table.index, pf.chain_table['gene']))
        lengths = dict(zip(pf.chain_table.index, pf.chain_table['length']))
    rows = []

    mhc_class = 'II' if 'B' in present else 'I' if 'A' in present else None
    if mhc_class is None:
        rows.append(_row('MHC class', 'fail', 'no MHC chain found'))
    else:
        rows.append(_row('MHC class', 'ok', f'class {mhc_class}: ' + ', '.join(
            genes[c] for c in ('A', 'B', 'F') if c in genes)))
        missing = [c for c in _required(mhc_class, kind) if c not in present]
        b2m_ok = not require_b2m and missing == ['F']
        if missing and not b2m_ok:
            missing_roles = [f'{c} ({_ROLE[c]})' for c in missing]
            rows.append(_row('chains', 'fail', 'missing ' + ', '.join(missing_roles)))
        else:
            rows.append(_row('chains', 'warning' if b2m_ok else 'ok',
                             'B2M missing (require_b2m=True makes this a failure)' if b2m_ok else
                             'all present: ' + ', '.join(sorted(present & set('ABCDEF')))))

    if 'C' in present:
        n = lengths['C']
        lo, hi = _PEPTIDE_LENGTH.get(mhc_class, (8, 30))
        rows.append(_row('peptide', 'ok' if lo <= n <= hi else 'warning',
                         f'{n} residues' + ('' if lo <= n <= hi else
                                            f' (usual for class {mhc_class}: {lo}-{hi})')))
    else:
        fused = find_fused_peptide(structure if is_seq else pf)
        if len(fused):
            f = fused.iloc[0]
            rows.append(_row('peptide', 'fail',
                             f'fused to chain {f["chain"]} ({f["side"]}-terminal, '
                             f'{f["start"]}-{f["end"]}: {f["peptide"]}; evidence: '
                             f'{f["evidence"]}); split it with tcs.split_fused_peptide()'))
        else:
            rows.append(_row('peptide', 'fail', 'no peptide chain, and none fused to another'))

    meta = {} if is_seq else (pf.metadata or {})
    if 'complex' in meta:
        n = meta['n_complexes']
        rows.append(_row('copies', 'warning',
                         f'complex {meta["complex"]} of {n} in the file (by contacts), file '
                         f'chains {", ".join(meta["complex_chains"])}; complex=n checks '
                         'another, tcs.complex_table() shows how the chains group'
                         if n > 1 else f'one complex (file chains '
                         f'{", ".join(meta["complex_chains"])}) and chains in no complex, '
                         'left out; tcs.complex_table() lists them'))
    extra = sorted(present - set('ABCDEF'))
    if extra:
        rows.append(_row('extra chains', 'warning',
                         f'{len(extra)} more chains ({", ".join(extra)}): ' +
                         ('other copies in the asymmetric unit, or other molecules; A-F are '
                          'the first chain of each kind by chain ID' if complex is None
                          else 'other molecules, not part of the complex')))
        if complex is None and not is_seq and 'original_chain' in pf.df and 'A' in present:
            rows.append(_one_complex(pf))

    if contacts and not is_seq:
        rows += _contacts(pf, kind, cutoff)
    if structure_checks and not is_seq:
        rows.append(_breaks(pf))
        if pf.mhc_class:
            rows.append(_mhc_anchors(pf))
        if {'D', 'E'} & present:
            rows += [_anchors(pf), _numbering(pf)]
    out = pd.DataFrame(rows, columns=['check', 'status', 'detail'])
    out.attrs['valid'] = bool((out['status'] != 'fail').all())
    out.attrs['name'] = name
    return out.attrs['valid'] if as_bool else out


def is_valid(structure, **kwargs) -> bool:
    """Whether :func:`verify` finds nothing failing (``verify(..., as_bool=True)``)."""
    return verify(structure, as_bool=True, **kwargs)


def _verify_one(args) -> tuple[dict, list[str]]:
    from .batch import _recorded
    path, kwargs = args
    try:
        table, caught = _recorded(verify, path, **kwargs)
    except Exception as e:
        return {'file': str(path), 'valid': False, 'failed': f'error: {type(e).__name__}: {e}',
                'warned': ''}, []

    def pick(status):
        return '; '.join(f'{r.check}: {r.detail}' for r in table.itertuples()
                         if r.status == status)

    return {'file': str(path), 'valid': table.attrs['valid'], 'failed': pick('fail'),
            'warned': pick('warning')}, caught


def verify_dir(directory: str | Path, n_jobs: int = 1, progress: bool = False,
               **kwargs) -> pd.DataFrame:
    """:func:`verify` every structure under `directory`: one row per file.

    Returns:
        ``file``, ``valid``, ``failed`` and ``warned`` (the failing / warning checks
        with their details). `kwargs` go to :func:`verify`.
    """
    from .batch import _map, _replay, find_structures
    files = find_structures(directory)
    results = _map(_verify_one, [(f, kwargs) for f in files], n_jobs, progress, 'verifying')
    for f, (_, caught) in zip(files, results):
        _replay(f.name, caught)
    return pd.DataFrame([row for row, _ in results], columns=['file', 'valid', 'failed', 'warned'])
