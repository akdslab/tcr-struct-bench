"""Peptides fused to another chain, and splitting them into their own chain.

Some constructs tie the peptide to a protein chain with a linker: to the MHC beta
chain (common for class II, e.g. 6XCP), to B2M or the heavy chain (class I
single-chain trimers), or to a TCR chain. In the file the peptide is then part of
that chain, so no peptide chain is detected.

    >>> tcs.find_fused_peptide(pf)          # where a fused peptide is, one row each
    >>> pdb1 = tcs.split_fused_peptide(pf)  # the peptide moved to its own chain, C

How a fused peptide is found, in a structure:

1. **Overhangs.** The residues of each chain outside its own domain: for MHC and B2M
   chains, those outside the local alignment to the best-matching MHC sequence
   (:func:`tcrkit.mhc.mhc_span`); for TCR chains, those before the ANARCI-numbered V
   domain.
2. **Groove contact.** An overhang residue is in the groove when one of its heavy
   atoms is within `cutoff` (4.5 Å) of the MHC groove helices.
3. **The peptide** runs from the first to the last groove residue of an overhang, and
   needs at least 7 groove residues. The rest of the overhang (usually a linker, often
   partly disordered and missing) is the linker.

A peptide is only called when it really lies in the groove, so tags and linkers
are not mistaken for one. With sequences only (no coordinates), see
:func:`find_fused_peptide`: the call is weaker, and marked as such.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure import deposited
from pdbframe.structure.geometry import atom_pairs_within
from tcrkit.mhc import mhc_span

from ._warnings import warn
from .chains import annotate

if TYPE_CHECKING:
    from .tcr import TCRFrame

__all__ = ['find_fused_peptide', 'split_fused_peptide']

_MIN_GROOVE = 7
_COLUMNS = ['chain', 'side', 'start', 'end', 'peptide', 'linker', 'groove_residues', 'evidence']
_LINKER = re.compile(r'[GS]{3,}[GSA]*$')


def _label(row) -> str:
    return f'{row.residue_number}{row.insertion_code}'


def _overhangs(pf: PDBFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """``(chain, 'N' or 'C', residue rows)`` for each chain's residues outside its domain."""
    res = deposited.observed_residues(pf.df).merge(pf.df.drop_duplicates(['chain', 'observed_index'])
                            [['chain', 'observed_index', 'imgt']], on=['chain', 'observed_index'])
    out = []
    for chain, g in res.groupby('chain', sort=False):
        g = g.reset_index(drop=True)
        seq = ''.join(g['aa'])
        if chain in ('A', 'B', 'F'):
            span = mhc_span(seq)
            if span is None:
                continue
            start, end = span
        elif chain in ('D', 'E'):
            leader = g['imgt'].str.match(r'^-\d')
            start, end = int(leader.sum()), len(g)
        else:
            continue
        if start > 0:
            out.append((chain, 'N', g.iloc[:start]))
        if end < len(g):
            out.append((chain, 'C', g.iloc[end:]))
    return out


def _groove_atoms(pf: PDBFrame) -> np.ndarray:
    from .mhc import mhc_helices
    try:
        mask = mhc_helices(pf)
        if mask.any():
            return mask
    except Exception as e:
        warn(f'{pf.name}: MHC helices not found ({e}); using the whole MHC to find '
             'the groove', stacklevel=5)
    return pf.df['chain'].isin(['A', 'B']).to_numpy()


def _from_structure(pf: PDBFrame, cutoff: float) -> pd.DataFrame:
    groove = _groove_atoms(pf)
    labels = (pf.df['chain'] + ':' + pf.df['observed_index'].astype(str)).to_numpy()
    rows = []
    for chain, side, over in _overhangs(pf):
        keys = chain + ':' + over['observed_index'].astype(str)
        atoms = np.isin(labels, keys.to_numpy()) & ~groove
        if not atoms.any():     # an overhang can be sequence only, with no atoms
            continue
        i, _, _ = atom_pairs_within(pf.coords[atoms], pf.coords[groove], cutoff)
        near = set(labels[atoms][i])
        in_groove = keys.isin(near).to_numpy()
        if in_groove.sum() < _MIN_GROOVE:
            continue
        first, last = np.flatnonzero(in_groove)[[0, -1]]
        pep = over.iloc[first:last + 1]
        link = over.iloc[last + 1:] if side == 'N' else over.iloc[:first]
        rows.append({'chain': chain, 'side': side,
                     'start': _label(pep.iloc[0]), 'end': _label(pep.iloc[-1]),
                     'peptide': ''.join(pep['aa']), 'linker': ''.join(link['aa']),
                     'groove_residues': int(in_groove.sum()), 'evidence': 'structure',
                     '_peptide_index': pep['observed_index'].tolist(),
                     '_linker_index': link['observed_index'].tolist()})
    return pd.DataFrame(rows, columns=_COLUMNS + ['_peptide_index', '_linker_index'])


def _from_sequences(seqs: dict[str, str]) -> pd.DataFrame:
    from .chains import detect_chains
    table = detect_chains(seqs)
    rows = []
    for chain, row in table.iterrows():
        if row['type'] != 'MHC':
            continue
        seq = row['sequence']
        span = mhc_span(seq)
        if span is None or span[0] < _MIN_GROOVE:
            continue
        head = seq[:span[0]]
        linker = _LINKER.search(head)
        pep = head[:linker.start()] if linker else head
        if len(pep) >= _MIN_GROOVE:
            rows.append({'chain': chain, 'side': 'N', 'start': '1', 'end': str(len(pep)),
                         'peptide': pep, 'linker': head[len(pep):],
                         'groove_residues': None, 'evidence': 'sequence (possible)'})
    return pd.DataFrame(rows, columns=_COLUMNS)


def find_fused_peptide(structure, cutoff: float = 4.5) -> pd.DataFrame:
    """Peptides fused to another chain: one row per candidate (usually none or one).

    Args:
        structure: a file or PDBFrame, or just sequences ``{chain: sequence}``.
        cutoff: distance (Å) to the MHC helices that counts as lying in the groove.

    Returns:
        A table with the ``chain`` it is fused to (standard IDs for structures, the
        given IDs for sequences), its ``side`` (``'N'`` / ``'C'``), the ``start`` and
        ``end`` residue numbers (positions in the sequence, for sequences), the
        ``peptide`` and ``linker`` sequences, the number of ``groove_residues``, and
        the ``evidence``: ``'structure'`` (it lies in the groove) or ``'sequence
        (possible)'`` (an N-terminal overhang before the MHC domain, minus a Gly/Ser
        linker; worth a look, not proof).
    """
    if isinstance(structure, dict):
        return _from_sequences(structure)
    return _from_structure(annotate(structure), cutoff)[_COLUMNS]


def split_fused_peptide(structure, cutoff: float = 4.5, linker: str = 'keep') -> TCRFrame:
    """The structure with a fused peptide moved into its own chain, C.

    Finds the peptide with :func:`find_fused_peptide`, then gives its residues chain
    ID C (they keep their residue numbers; their ``imgt`` label becomes ``'-'``).

    Args:
        structure: a file or PDBFrame.
        linker: ``'keep'`` leaves the linker residues with the chain they were in;
            ``'drop'`` removes them.

    Returns:
        A :class:`~tcrstruct.TCRFrame`. A structure without a fused
        peptide comes back unchanged, with a warning.
    """
    if linker not in ('keep', 'drop'):
        raise ValueError(f"linker must be 'keep' or 'drop', not {linker!r}")
    pf = annotate(structure)
    found = _from_structure(pf, cutoff)
    if found.empty:
        warn(f'{pf.name}: no fused peptide found; structure unchanged', stacklevel=3)
        return pf
    if len(found) > 1:
        raise ValueError(f'{pf.name}: {len(found)} fused peptide candidates '
                         f'({", ".join(found["chain"])}); see find_fused_peptide()')
    if 'C' in pf.chains:
        raise ValueError(f'{pf.name} already has a peptide chain C; the overhang of chain '
                         f'{found["chain"].iloc[0]} may be a second copy of the peptide')
    hit = found.iloc[0]
    df = pf.df.copy()
    on_chain = (df['chain'] == hit['chain']).to_numpy()
    pep = on_chain & df['observed_index'].isin(hit['_peptide_index']).to_numpy()
    if linker == 'drop':
        keep = ~(on_chain & df['observed_index'].isin(hit['_linker_index']).to_numpy())
        df, pep = df[keep].reset_index(drop=True), pep[keep]
    df.loc[pep, 'chain'] = 'C'
    df.loc[pep, 'imgt'] = '-'
    df = df.iloc[np.argsort(df['chain'].to_numpy(), kind='stable')]
    return annotate(PDBFrame(df, name=pf.name))
