"""The MHC side: gene, IMGT numbering of the G domains, and the groove helices.

All of it is sequence work and comes from one tcrkit call per MHC chain,
:func:`tcrkit.annotate_mhc` (cached by sequence, so a chain is annotated once): its
gene and closest allele, and each MHC groove domain (G-DOMAIN) numbered on IMGT's
scheme, strands A-D (1-49) and the helix (50-92B), with insertions such as ``'54A'``.
Class I heavy chains have two G domains, G-ALPHA1 and G-ALPHA2; class II has
G-ALPHA on the alpha chain and G-BETA on the beta chain. This module puts it on
a structure's chains A / B.

    >>> mhc_numbering(pf)                       # per residue of chains A / B
    >>> mhc_helices(pf)                         # the groove helices, by IMGT position
    >>> mhc_helices(pf, method='v1')            # the 'v1' helices (MHC Motif Atlas)
    >>> mhc_match(seq)                          # gene, closest allele and its identity
    >>> tcrkit.annotate_mhc(seq)                # gene and numbering of a bare sequence

:func:`.chains.annotate` writes the positions into the atom table's ``imgt``
column, as for the TCR chains. Class I chain A has two G domains, each numbered
from 1, so its positions repeat; :func:`mhc_numbering` says which domain each
residue is in. Residues outside the G domains are ``'-'``, except a signal peptide,
numbered ``-k`` ... ``-1`` back from IMGT 1 (as tcrkit does).

Helix residues, by method (both refer to the same helix; see
``tests/test_mhc.py``):

- ``'imgt'`` (default): fixed IMGT positions
  (:data:`tcrkit.gdomain.HELIX`), helix 50-86 (G-ALPHA1, G-ALPHA), 52-86
  (G-ALPHA2) and 52-61A plus 63-86 (G-BETA). They are the Pierce lab's residues,
  HLA-A2 50-86 and 140-176, HLA-DR alpha 46-78 and beta 54-64 plus 67-91
  (:data:`PIERCE_HELICES`), placed the same way in every allele. They are part of
  IMGT's helix region, not all of it.
- ``'v1'``: the MHC Motif Atlas (http://mhcmotifatlas.org) ranges of the
  closest gene (tcrkit's Motif Atlas helices). In IMGT
  numbering they start gene by
  gene at 49-52 on class II alpha and 52-56 on beta.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure import deposited
from tcrkit import annotate_mhc
from tcrkit.gdomain import GDOMAINS

from ._warnings import warn
from .numbering import full_sequence

__all__ = ['mhc_annotation', 'mhc_match', 'mhc_numbering', 'mhc_chains', 'GDOMAINS',
           'is_class_ii', 'mhc_helices', 'HELIX_METHODS', 'PIERCE_HELICES']

#: Ways to pick the groove helices; see :func:`mhc_helices`.
HELIX_METHODS = ('imgt', 'v1')

#: tcrkit's name for each method (``annotate_mhc(helix=...)``).
_TCRKIT_HELIX = {'imgt': 'imgt', 'v1': 'tpm-bench'}  # tcrkit's API name for the Motif Atlas helices

#: Helix residues of the Pierce lab's ``tcr_docking_angle``, in standard MHC numbering
#: (HLA-A2 for class I, HLA-DR for class II): ``{class: {chain: [(first, last), ...]}}``.
#: tcrstruct finds them by their IMGT positions (:data:`tcrkit.gdomain.HELIX`).
PIERCE_HELICES = {'I': {'A': [(50, 86), (140, 176)]},
                  'II': {'A': [(46, 78)], 'B': [(54, 64), (67, 91)]}}


def is_class_ii(pf: PDBFrame) -> bool:
    """Class II complexes have an MHC beta chain, B."""
    return 'B' in pf.chains


def mhc_chains(pf: PDBFrame) -> list[str]:
    """The chains with G domains: A, and B for class II."""
    return [c for c in (('A', 'B') if is_class_ii(pf) else ('A',)) if c in pf.chains]


# ------------------------------------------------------------ IMGT numbering

@lru_cache(maxsize=1024)
def _annotated(sequence: str, helix: str) -> pd.DataFrame:
    return annotate_mhc(sequence, helix=helix, per_residue=True).reset_index(drop=True)


def mhc_annotation(sequence: str, helix: str = 'imgt') -> pd.DataFrame:
    """tcrkit's per-residue annotation of one MHC chain (:func:`tcrkit.annotate_mhc`
    with ``per_residue=True``): ``gene``, ``allele``, ``domain``, ``imgt``,
    ``region``, ``helix`` (by `helix`, see :data:`HELIX_METHODS`), ``topology``.
    Cached by sequence; a copy is returned."""
    return _annotated(sequence, helix).copy()


@lru_cache(maxsize=1024)
def _matched(sequence: str) -> pd.Series:
    return annotate_mhc(sequence).iloc[0]


def mhc_match(sequence: str) -> pd.Series:
    """tcrkit's one-row summary of an MHC (or B2M) chain (:func:`tcrkit.annotate_mhc`):
    ``gene``, ``allele``, ``allele_identity`` (identical residues over the aligned
    part; 1.0 is an exact match), ``mhc_class``, ``domains``, ``helix``, the topology
    regions, ... Cached by sequence; a copy is returned."""
    return _matched(sequence).copy()


def _on_residues(pf: PDBFrame, chain: str, annotate) -> pd.DataFrame:
    """``annotate(sequence)``, one row per residue, for the chain's residues with atoms:
    run on its deposited sequence when the file had one (:func:`.numbering.full_sequence`),
    so a gap does not shift the numbering, else on the residues it has."""
    try:
        full = full_sequence(pf, chain)
    except ValueError:            # annotate() has warned about this chain
        full = None
    if full is None:
        return annotate(pf.sequences[chain])
    return annotate(full[0]).iloc[full[1]].reset_index(drop=True)


def mhc_numbering(pf: PDBFrame) -> pd.DataFrame:
    """IMGT G-domain numbering of an annotated structure's MHC chains (A, and B for class II).

    Returns:
        One row per residue of those chains: ``chain``, ``observed_index``,
        ``residue_number``, ``insertion_code`` and tcrkit's annotation
        (:func:`mhc_annotation`): ``gene``, ``allele``, ``domain``, ``imgt``,
        ``region``, ``helix`` (the ``'imgt'`` helices), then ``aa`` and ``topology``
        (``'extracellular'``, ``'signal'``, ``'transmembrane'``, ``'cytoplasmic'`` or
        ``'-'``, from the matched allele).
    """
    res = deposited.observed_residues(pf.df)     # residues with atoms
    cols = ['chain', 'observed_index', 'residue_number', 'insertion_code', 'gene', 'allele',
            'domain', 'imgt', 'region', 'helix', 'aa', 'topology']
    parts = []
    for chain in mhc_chains(pf):
        num = _on_residues(pf, chain, mhc_annotation)
        rows = res[res['chain'] == chain][cols[:4]].reset_index(drop=True)
        parts.append(pd.concat([rows, num[cols[4:]]], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


# ------------------------------------------------------------- groove helices

def mhc_helices(pf: PDBFrame, chains=('A', 'B'), method: str = 'imgt') -> np.ndarray:
    """Atoms of the MHC groove helices (both helices of class I chain A, or the
    alpha helix of A and the beta helix of B for class II).

    Args:
        chains: the MHC chains to take helices from.
        method: ``'imgt'`` (default): fixed IMGT positions
            (:data:`tcrkit.gdomain.HELIX`), the Pierce lab's residues
            (:data:`PIERCE_HELICES`) in any allele and any file numbering.
            ``'v1'``: each MHC chain is matched
            to its closest MHC Motif Atlas gene (http://mhcmotifatlas.org; chains
            under 30 residues are not matched), and the gene's helix ranges are
            carried over by sequence alignment. Same as ``'imgt'`` on class I; on
            class II the ends differ by a few residues. Both come from
            :func:`tcrkit.annotate_mhc` (``'v1'``: its Motif Atlas helices).
    """
    if method not in HELIX_METHODS:
        raise ValueError(f'helix method must be one of {HELIX_METHODS}, got {method!r}')
    res = deposited.observed_residues(pf.df)     # residues with atoms
    picked = []
    for chain in [c for c in mhc_chains(pf) if c in chains]:
        seq = pf.sequences[chain]
        if method == 'v1' and len(seq) < 30:
            raise ValueError(f'chain {chain} is too short ({len(seq)}) to match an MHC gene')
        helix = _on_residues(pf, chain, lambda s: mhc_annotation(s, _TCRKIT_HELIX[method])
                             )['helix'].to_numpy(bool)
        if not helix.any():
            other = 'v1' if method == 'imgt' else 'imgt'
            warn(f'{pf.name}: chain {chain}: no groove-helix residues, as no IMGT G domain '
                 f"was numbered; try mhc_helices(..., method='{other}')", stacklevel=3)
        index = res.loc[res['chain'] == chain, 'observed_index'].to_numpy()
        picked += [(chain, i) for i in index[helix]]
    keys = pd.MultiIndex.from_frame(pf.df[['chain', 'observed_index']])
    return keys.isin(pd.MultiIndex.from_tuples(picked, names=['chain', 'observed_index'])
                     if picked else [])
