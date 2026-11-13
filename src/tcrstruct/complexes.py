"""Several TCR-pMHC complexes in one file: find each, by chain type and contacts.

    >>> complexes = find_complexes('2ckb.cif')       # one TCRFrame per complex
    >>> complexes[1].chain_table                     # the second copy, chains A-F
    >>> complex_table('2ckb.cif')                    # which file chains make up each
    >>> TCRFrame.from_rcsb('2CKB', complex=2)        # the same as complexes[1]

An asymmetric unit (the deposited RCSB entry) often holds two or more copies of a
complex. Reading such a file (:meth:`TCRFrame.read`, :func:`.chains.annotate`) keeps
one of them, complex 1 by default (``complex=n`` for another, ``complex=None`` for
every chain, the first of each kind by chain ID getting its standard ID, which can mix
copies); this module finds them. The chains are typed as in
:func:`.chains.detect_chains` and grouped by contacts (Cα atoms within 8 Å):

1. each MHC groove chain (class I heavy chain, class II alpha), by chain ID,
   starts a complex;
2. it takes the free B2M (class I) or class II beta chain it touches most;
3. then the free peptide that touches that MHC most;
4. then the free TCR alpha that touches the pMHC most, and the free TCR beta that
   touches that alpha most (or, without an alpha, the beta that touches the pMHC
   most).

A chain joins only with at least `min_contacts` contacting Cα pairs. Chains left
over (a TCR on no MHC, a lone B2M, ...) are named in a warning.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure import deposited
from scipy.spatial import cKDTree
from tcrkit.mhc import mhc_chain

from ._warnings import warn

if TYPE_CHECKING:
    from .tcr import TCRFrame

__all__ = ['find_complexes', 'complex_table', 'chain_contacts']


def _plain(structure) -> PDBFrame:
    """`structure` as a plain PDBFrame with the file's chain IDs (an annotated one is
    turned back)."""
    pf = PDBFrame.ensure(structure)
    df = pf.df
    if 'original_chain' not in df:
        return PDBFrame._wrap(df, pf.name, pf.metadata, pf._seqres)
    df = df.assign(chain=df['original_chain']).drop(columns=['original_chain', 'imgt'],
                                                    errors='ignore')
    seqres = deposited.remap(pf._seqres, pf.df, df)
    out = PDBFrame(df, name=pf.name, metadata=pf.metadata)
    out._seqres = None if seqres is None else seqres.drop(columns=['imgt'], errors='ignore')
    return out


def chain_contacts(pf: PDBFrame, cutoff: float = 8.0) -> pd.DataFrame:
    """Contacting Cα pairs (within `cutoff` Å) between every two chains: a symmetric
    chain x chain table of counts."""
    df = pf.df
    ca = df[df['atom_name'] == 'CA']
    chains = list(dict.fromkeys(df['chain']))
    out = pd.DataFrame(0, index=chains, columns=chains)
    if len(ca) < 2:
        return out
    pairs = cKDTree(ca[['x', 'y', 'z']].to_numpy(float)).query_pairs(cutoff, output_type='ndarray')
    owner = ca['chain'].to_numpy()
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    between = a != b
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([a[between], b[between]])
                       ).groupby(level=[0, 1]).sum()
    for (x, y), n in counts.items():
        out.at[x, y] += n
        out.at[y, x] += n
    return out


def _kinds(table: pd.DataFrame) -> dict[str, str]:
    """Each chain's kind: ``'groove'`` (class I heavy / class II alpha), ``'B2M'``,
    ``'beta'`` (class II), ``'peptide'``, ``'alpha'`` / ``'tcr_beta'`` (TCR) or ``'other'``."""
    out = {}
    for c, row in table.iterrows():
        if row['type'] == 'MHC':
            k = mhc_chain(row['details'])
            out[c] = {'heavy': 'groove', 'alpha': 'groove', 'B2M': 'B2M', 'beta': 'beta'}.get(
                k, 'other')
        elif row['type'] == 'peptide':
            out[c] = 'peptide'
        elif row['type'] == 'TCR':
            out[c] = {'Alpha': 'alpha', 'D': 'alpha', 'Beta': 'tcr_beta'}.get(row['details'],
                                                                              'other')
        else:
            out[c] = 'other'
    return out


def complex_table(structure, min_contacts: int = 3, cutoff: float = 8.0,
                  _table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Which of the file's chains make up each complex (see the module docstring).

    Returns:
        One row per complex, numbered from 1 (the index ``complex``), with the
        file's chain ID of each part - ``mhc``, ``mhc_partner`` (B2M or class II
        beta), ``peptide``, ``tcr_alpha``, ``tcr_beta`` (``'-'`` if none) - and
        ``chains``, the list of them. ``result.attrs['unused']`` lists the chains in
        no complex.
    """
    from .chains import detect_chains
    pf = _plain(structure)
    table = detect_chains(pf) if _table is None else _table
    kinds = _kinds(table)
    touch = chain_contacts(pf, cutoff)
    used: set[str] = set()

    def take(kind: str, to: list[str]) -> str | None:
        """The free chain of `kind` that touches the chains `to` most, marked used."""
        free = [c for c, k in kinds.items() if k == kind and c not in used]
        if not free:
            return None
        score = {c: int(touch.loc[c, to].sum()) for c in free}
        c = max(free, key=lambda x: score[x])         # first by chain ID on ties
        if score[c] < min_contacts:
            return None
        used.add(c)
        return c

    rows = []
    for groove in [c for c, k in kinds.items() if k == 'groove']:
        used.add(groove)
        cls_ii = mhc_chain(table.at[groove, 'details']) == 'alpha'
        partner = take('beta' if cls_ii else 'B2M', [groove])
        mhc = [c for c in (groove, partner) if c]
        peptide = take('peptide', mhc)
        pmhc = [c for c in (*mhc, peptide) if c]
        alpha = take('alpha', pmhc)
        beta = take('tcr_beta', [alpha] if alpha else pmhc)
        parts = {'mhc': groove, 'mhc_partner': partner, 'peptide': peptide,
                 'tcr_alpha': alpha, 'tcr_beta': beta}
        rows.append({**{k: v or '-' for k, v in parts.items()},
                     'chains': [c for c in parts.values() if c]})
    out = pd.DataFrame(rows, columns=['mhc', 'mhc_partner', 'peptide', 'tcr_alpha',
                                      'tcr_beta', 'chains'])
    out.index = pd.RangeIndex(1, len(out) + 1, name='complex')
    out.attrs['unused'] = [c for c in kinds if c not in used]
    return out


def _has_copies(table: pd.DataFrame) -> bool:
    """Whether a :func:`.chains.detect_chains` table has two chains of one kind (two
    MHC groove chains, two peptides, two TCR alphas, ...)."""
    counts = Counter(k for k in _kinds(table).values() if k != 'other')
    return any(n > 1 for n in counts.values())


def _pick_complex(pf: PDBFrame, table: pd.DataFrame, complex: int, name: str
                  ) -> tuple[PDBFrame, dict]:
    """The chains of complex number `complex` (from 1) of `pf`, and what to add to its
    metadata; `pf` itself if it has one copy of each chain kind."""
    if complex < 1:
        raise ValueError(f'{name}: complex={complex}; complexes are numbered from 1 '
                         '(complex=None keeps every chain)')
    if not _has_copies(table):
        if complex != 1:
            raise ValueError(f'{name}: complex={complex}, but it has one complex '
                             '(numbered from 1); an RCSB biological assembly often holds '
                             'one copy: from_rcsb(..., assembly=None) reads the deposited '
                             'structure with every copy')
        return pf, {}
    groups = complex_table(pf, _table=table)
    if groups.empty:
        warn(f'{name}: two or more chains of one kind but no MHC groove chain (class I '
             'heavy chain or class II alpha) to group them by, so every chain is kept and '
             'the first of each kind by chain ID gets its standard ID; select the chains '
             'you want first (pf.select(chains=...))')
        return pf, {}
    if complex > len(groups):
        raise ValueError(f'{name}: complex={complex}, but it has {len(groups)} '
                         f'complex{"es" if len(groups) != 1 else ""} (numbered from 1)')
    chains = list(groups.at[complex, 'chains'])
    left_out = [c for c in table.index if c not in chains]
    if left_out:
        copies = (f'{len(groups)} copies of the complex (by contacts)' if len(groups) > 1
                  else 'one complex (by contacts) and chains in no complex')
        warn(f'{name}: {copies}; kept complex {complex} (file chains {", ".join(chains)}) and '
             f'left out chains {", ".join(left_out)}; complex=n reads another copy, '
             'complex=None every chain, tcs.complex_table() shows how they group')
    meta = {'complex': complex, 'complex_chains': chains, 'n_complexes': len(groups)}
    return pf.select(chains=chains), meta


def find_complexes(structure, min_contacts: int = 3, cutoff: float = 8.0) -> list[TCRFrame]:
    """Every TCR-pMHC (or pMHC) complex in a structure, each as an annotated TCRFrame.

    Args:
        structure: a file, PDBFrame or TCRFrame (a TCRFrame is taken back to the
            file's chain IDs first).
        min_contacts: the fewest contacting Cα pairs for a chain to join a complex.
        cutoff: the Cα-Cα distance (Å) that counts as a contact.

    Returns:
        One TCRFrame per complex, in the order of :func:`complex_table`, named
        ``'<name>_1'``, ``'<name>_2'``, ...; each has the standard chain IDs, and its
        ``metadata['complex']`` and ``metadata['complex_chains']`` (the file's
        chain IDs). Chains in no complex are named in a :class:`TCRStructWarning`.
    """
    from .chains import detect_chains
    from .tcr import TCRFrame
    pf = _plain(structure)
    table = detect_chains(pf)
    groups = complex_table(pf, min_contacts, cutoff, _table=table)
    unused = groups.attrs['unused']
    if groups.empty:
        warn(f'{pf.name}: no MHC groove chain (class I heavy chain or class II alpha) found, '
             'so no complex; annotate the structure as it is with TCRFrame(...) instead')
        return []
    if unused:
        listed = ', '.join(f'{c} ({table.at[c, "type"]} {table.at[c, "details"]})'
                           .replace(' -)', ')') for c in unused)
        warn(f'{pf.name}: chains in no complex: {listed}; they touch no MHC groove chain with '
             f'at least {min_contacts} Cα contacts (lower min_contacts= to include them)')
    out = []
    for number, row in groups.iterrows():
        part = pf.select(chains=row['chains'])
        meta = {**(pf.metadata or {}), 'complex': int(number),
                'complex_chains': list(row['chains']), 'n_complexes': len(groups)}
        frame = TCRFrame(PDBFrame._wrap(part.df, f'{pf.name}_{number}', meta, part._seqres))
        out.append(frame)
    return out
