"""Regions of an annotated TCR-pMHC structure, as boolean atom masks.

Every function takes a PDBFrame with the standard chain IDs (see
:func:`.chains.annotate`) and returns one boolean per atom of ``pf.df``, so masks
combine with ``&`` / ``|`` and select with ``pf.select(mask=...)``.

    >>> core = peptide_core(pf)                 # the peptide, or its 9-mer core (class II)
    >>> centric = near_residues(pf, core, 10)    # whole residues within 10 Å of it
    >>> cdr(pf, 'CDRB3')                         # IMGT 105-117 of TCR beta (chain E)
    >>> v_domain(pf, 'D')                        # IMGT 1-128 of TCR alpha
    >>> mhc_helices(pf)                          # the groove helices of chains A / B (Pierce)
    >>> mhc_helices(pf, method='v1')             # the 'v1' helices: Motif Atlas ranges

The MHC helices (:func:`mhc_helices`) live in :mod:`.mhc`, with the IMGT
numbering of the MHC; they are importable from here too.

Residues are told apart by ``(chain, observed_index)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure.geometry import atom_pairs_within

# the MHC helices moved to .mhc; imported here so tcs.regions.mhc_helices keeps working
from .mhc import HELIX_METHODS, PIERCE_HELICES, is_class_ii, mhc_helices
from .numbering import CDR_IMGT, imgt_position

__all__ = ['residue_labels', 'peptide_core', 'near_residues', 'cdr', 'v_domain',
           'unobserved_regions', 'CDRS',
           'is_class_ii', 'mhc_helices', 'HELIX_METHODS', 'PIERCE_HELICES']

#: The six CDRs, alpha (chain D) then beta (chain E).
CDRS = ['CDRA1', 'CDRA2', 'CDRA3', 'CDRB1', 'CDRB2', 'CDRB3']
_TCR_CHAIN = {'A': 'D', 'B': 'E'}

_CORE_LENGTH = 9


def residue_labels(df: pd.DataFrame) -> np.ndarray:
    """One label per atom naming its residue, unique across chains."""
    return (df['chain'] + ':' + df['observed_index'].astype(str)).to_numpy()


def _whole_residues(df: pd.DataFrame, atom_mask: np.ndarray) -> np.ndarray:
    """Widen an atom mask to every atom of the residues it touches."""
    labels = residue_labels(df)
    return np.isin(labels, labels[atom_mask])


# ---------------------------------------------------------------- the peptide

def peptide_core(pf: PDBFrame) -> np.ndarray:
    """The peptide atoms the benchmark scores.

    Class I: the whole peptide (chain C). Class II: the 9 consecutive residues
    closest to the MHC - each residue's distance is its closest heavy atom to any
    atom of chains A and B, and the window with the smallest sum wins (the first
    one on ties).
    """
    df = pf.df
    pep = (df['chain'] == 'C').to_numpy()
    if not pep.any():
        raise KeyError('no peptide (chain C)')
    if not is_class_ii(pf):
        return pep
    if not pf.has_coordinates:
        raise ValueError('a class II peptide core is found by distance to the MHC, so it '
                         'needs coordinates: read the structure, not only its sequences')
    mhc = df['chain'].isin(['A', 'B']).to_numpy()
    xyz = pf.coords
    # closest MHC atom to each peptide atom, then to each peptide residue
    dist = np.sqrt(((xyz[pep][:, None, :] - xyz[mhc][None, :, :]) ** 2).sum(-1)).min(axis=1)
    per_res = pd.Series(dist).groupby(df.loc[pep, 'observed_index'].to_numpy(), sort=False).min()
    if len(per_res) < _CORE_LENGTH:
        raise ValueError(f'peptide has {len(per_res)} residues; the core needs {_CORE_LENGTH}')
    sums = np.convolve(per_res.to_numpy(), np.ones(_CORE_LENGTH), 'valid')
    start = int(np.argmin(sums))
    core = per_res.index[start:start + _CORE_LENGTH]
    return pep & df['observed_index'].isin(core).to_numpy()


def near_residues(pf: PDBFrame, mask: np.ndarray, cutoff: float = 10.0,
                  within: np.ndarray | None = None) -> np.ndarray:
    """Whole residues with any atom within `cutoff` Å of the atoms in `mask`.

    Args:
        within: only look among these atoms (default: all chains, the masked
            atoms' own residues included).
    """
    within = np.ones(len(pf.df), bool) if within is None else np.asarray(within, bool)
    xyz = pf.coords
    i, _, _ = atom_pairs_within(xyz[within], xyz[mask], cutoff)
    hit = np.zeros(len(xyz), bool)
    hit[np.flatnonzero(within)[i]] = True
    return _whole_residues(pf.df, hit)


# -------------------------------------------------------------------- the TCR

def cdr(pf: PDBFrame, name: str) -> np.ndarray:
    """Atoms of one CDR (``'CDRA1'`` ... ``'CDRB3'``), by the frame's own IMGT labels.

    IMGT ranges are inclusive (CDR1 27-38, CDR2 56-65, CDR3 105-117) and an
    insertion counts as its position (111A is 111).
    """
    if name not in CDRS:
        raise ValueError(f'unknown CDR {name!r}; use one of {CDRS}')
    lo, hi = CDR_IMGT[int(name[-1])]
    return _imgt_range(pf, _TCR_CHAIN[name[3]], lo, hi)


def v_domain(pf: PDBFrame, chain: str) -> np.ndarray:
    """Atoms of a TCR chain's V domain (``'D'`` alpha, ``'E'`` beta): IMGT 1-128, by
    the frame's own labels. It holds the three CDRs; the constant domain is left out."""
    if chain not in ('D', 'E'):
        raise ValueError(f"V domains are on the TCR chains 'D' and 'E', not {chain!r}")
    return _imgt_range(pf, chain, 1, 128)


def unobserved_regions(pf: PDBFrame) -> dict[str, np.ndarray]:
    """``{region: mask}`` over the deposited residues of `pf` (PDBFrame's ``pf._seqres``,
    behind ``pf.residues``), for the regions a residue without atoms is in by its labels:
    the CDRs and V domains (by IMGT) and the peptide. The others (peptide core, MHC
    helices, peptide-centric) are found from atoms. Empty without a deposited sequence.
    """
    seqres = getattr(pf, '_seqres', None)
    if seqres is None:
        return {}
    imgt = seqres['imgt'] if 'imgt' in seqres else pd.Series('-', index=seqres.index)
    pos = imgt.map(lambda s: imgt_position(s) if s[:1] != '-' else None)
    pos = pos.astype(float).to_numpy()
    chain = seqres['chain'].to_numpy()
    out = {}
    with np.errstate(invalid='ignore'):
        for name in CDRS:
            lo, hi = CDR_IMGT[int(name[-1])]
            out[name] = (chain == _TCR_CHAIN[name[3]]) & (pos >= lo) & (pos <= hi)
        for name, c in [('Va', 'D'), ('Vb', 'E')]:
            out[name] = (chain == c) & (pos >= 1) & (pos <= 128)
    out['peptide'] = chain == 'C'
    return out


def _imgt_range(pf: PDBFrame, chain: str, lo: int, hi: int) -> np.ndarray:
    """Atoms of `chain` whose IMGT position (111A counts as 111) is in ``lo..hi``."""
    if 'imgt' not in pf.df:
        raise KeyError("no 'imgt' column; read the structure with TCRFrame.read() or "
                       'annotate it with tcrstruct.annotate()')
    pos = pf.df['imgt'].map(lambda s: imgt_position(s) if s[:1] != '-' else None)
    pos = pos.astype(float).to_numpy()
    on_chain = (pf.df['chain'] == chain).to_numpy()
    with np.errstate(invalid='ignore'):
        return on_chain & (pos >= lo) & (pos <= hi)
