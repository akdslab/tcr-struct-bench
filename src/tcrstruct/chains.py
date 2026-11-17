"""Finding the chains of a TCR-pMHC complex and giving them the standard IDs.

    >>> pf = annotate(PDBFrame.read('1ao7.pdb'))   # a TCRFrame: chains A-F, 'imgt' column
    >>> normalize('1ao7.pdb').write('1ao7_normalized.pdb')   # + TCRs numbered by IMGT
    >>> detect_chains(PDBFrame.read('1ao7.pdb'))     # the table behind the renaming

Chains are identified by these rules:

- **TCR**: chains ANARCI numbers (:mod:`.numbering`, every chain type allowed); type A
  is alpha, B is beta (D is taken as alpha when no A is found).
- **MHC / B2M**: the remaining chains of at least 30 residues whose best local
  alignment to tcrkit's MHC / B2M sequences (:func:`tcrkit.mhc.best_mhc`) scores more
  than 0.07 per residue (score divided by the longer sequence).
- **peptide**: the remaining chains of 1-27 standard residues.

Each TCR, MHC and B2M chain also gets a **species**: the MHC's is that of its closest
sequence in tcrkit (human, mouse and the other species of TCR3d's structures:
macaque, pig, cattle, chicken, ...), the TCR's that of its ANARCI germline genes
(human or mouse, the only TCR germlines ANARCI has).

and renamed (:data:`CHAIN_IDS`): A (MHC heavy / class II alpha), B (class II beta),
C (peptide), D (TCR alpha), E (TCR beta), F (B2M); any other chain becomes G, H, ...
A class I molecule needs its heavy chain or B2M, a class II one both of its chains.

A file with two chains of one kind (an asymmetric unit with several copies of the
complex) is cut down to one complex first, found by contacts (complex 1 by default,
``complex=n`` for another; see :mod:`.complexes`). With ``complex=None`` every chain
is kept, and of two chains of one kind the first by chain ID gets the standard ID.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure import deposited
from tcrkit.mhc import best_mhc, gene_domains, mhc_chain, mhc_species

from ._warnings import warn
from .mhc import mhc_annotation
from .numbering import Numbering, full_sequence, imgt_labels, run_anarci

if TYPE_CHECKING:
    from .tcr import TCRFrame

__all__ = ['annotate', 'is_annotated', 'normalize', 'detect_chains', 'heavy_atoms',
           'CHAIN_IDS']

#: The standard chain IDs :func:`detect_chains` gives.
CHAIN_IDS = {'A': 'MHC heavy chain (class I) or alpha chain (class II)',
             'B': 'MHC beta chain (class II)', 'C': 'peptide', 'D': 'TCR alpha',
             'E': 'TCR beta', 'F': 'B2M'}

#: The thresholds: an MHC chain has at least 30 residues and matches an MHC
#: sequence with more than 0.07 per residue; a peptide has at most 27 standard residues.
_MHC_MIN_LENGTH = 30
_MHC_MIN_SCORE = 0.07
_PEPTIDE_MAX_LENGTH = 27
_PEPTIDE_LETTERS = set('ACDEFGHIKLMNPQRSTVWYX')

# ------------------------------------------------------------------- detecting

def detect_chains(pf, numbering: dict[str, Numbering | None] | None = None
                  ) -> pd.DataFrame:
    """What each chain of a TCR-pMHC (or pMHC) structure is, and its standard ID.

    >>> detect_chains({'H': hla_a2, 'L': b2m, 'P': 'LLFGYPVYV', 'Q': alpha, 'R': beta})
    #   type  details  new_chain  species  sequence
    # H  MHC        A          A    human  GSHSMRYF...

    Args:
        pf: the structure (a PDBFrame), or just its sequences, ``{chain: sequence}``
            (a dict or Series).
        numbering: ANARCI results per chain, if already computed
            (see :func:`.numbering.run_anarci`); chains left out are not TCRs. By
            default ANARCI is run here.

    Returns:
        One row per chain, sorted by chain ID (the index: the chain's ID in `pf`):
        ``type`` (``'TCR'``, ``'MHC'``, ``'peptide'`` or ``'None'``), ``details``
        (``'Alpha'`` / ``'Beta'`` for TCRs, or ANARCI's chain type for other
        receptors, e.g. ``'D'``; the gene for MHC chains, e.g. ``'A'`` or ``'DRB1'``;
        ``'-'`` otherwise), ``new_chain`` (the standard ID, :data:`CHAIN_IDS`),
        ``species`` (of the closest MHC / B2M sequence, or of the TCR germline genes
        ANARCI assigns; ``'-'`` for other chains) and ``sequence``.
    """
    if isinstance(pf, pd.Series):
        pf = pf.to_dict()
    if isinstance(pf, dict):
        seqs = pf
    elif isinstance(pf, PDBFrame):
        seqs = pf.sequences
    else:
        raise TypeError('detect_chains takes a PDBFrame or its sequences as '
                        f'{{chain ID: sequence}}, got {type(pf).__name__}')
    # chains in sorted order: the "first" chain of a kind is by ID
    seqs = {str(c): str(s) for c, s in sorted(seqs.items())}
    numbering = run_anarci(seqs) if numbering is None else numbering
    tcr = {c: n.chain_type for c, n in numbering.items() if n is not None and c in seqs}

    mhc, species = {}, {c: numbering[c].species for c in tcr}
    for c, s in seqs.items():
        if c in tcr or len(s) < _MHC_MIN_LENGTH:
            continue
        gene, allele, score = best_mhc(s)
        if gene is not None and score > _MHC_MIN_SCORE:
            mhc[c] = gene
            species[c] = mhc_species(allele) or '-'
    peptide = [c for c, s in seqs.items() if c not in tcr and c not in mhc
               and 0 < len(s) <= _PEPTIDE_MAX_LENGTH and set(s.upper()) <= _PEPTIDE_LETTERS]

    tcr_type = {c: {'A': 'Alpha', 'B': 'Beta'}.get(t, t) for c, t in tcr.items()}
    table = pd.DataFrame({'sequence': pd.Series(seqs, dtype=object)})
    table['type'] = ['TCR' if c in tcr else 'MHC' if c in mhc else
                     'peptide' if c in peptide else 'None' for c in table.index]
    table['details'] = [tcr_type.get(c) or mhc.get(c) or '-' for c in table.index]
    table['new_chain'] = _standard_ids(table)
    table['species'] = [species.get(c, '-') for c in table.index]
    return table[['type', 'details', 'new_chain', 'species', 'sequence']]


def _standard_ids(table: pd.DataFrame) -> list[str]:
    """Standard chain IDs (:data:`CHAIN_IDS`) for the rows of a :func:`detect_chains` table."""
    new: dict[str, str] = {}

    def of(kind, test=lambda d: True) -> list[str]:
        """Chains of one type whose details pass `test`, in table order."""
        return [c for c in table.index
                if table.at[c, 'type'] == kind and test(table.at[c, 'details'])]

    mhc = {c: mhc_chain(table.at[c, 'details']) for c in of('MHC')}
    kind = {k: [c for c, v in mhc.items() if v == k] for k in ('heavy', 'B2M', 'alpha', 'beta')}
    if kind['heavy'] or kind['B2M']:                     # class I
        if kind['heavy']:
            new[kind['heavy'][0]] = 'A'
        if kind['B2M']:
            new[kind['B2M'][0]] = 'F'
    elif len(mhc) >= 2:                                  # class II needs both chains
        if kind['alpha']:
            new[kind['alpha'][0]] = 'A'
        if kind['beta']:
            new[kind['beta'][0]] = 'B'
    pep = of('peptide')
    if pep:
        new[pep[0]] = 'C'
    alpha = of('TCR', lambda d: d == 'Alpha') or of('TCR', lambda d: d == 'D')
    beta = of('TCR', lambda d: d == 'Beta')
    if alpha:
        new[alpha[0]] = 'D'
    if beta:
        new[beta[0]] = 'E'
    spare = (x for x in string.ascii_uppercase[6:] if x not in new.values())
    return [new.get(c) or next(spare) for c in table.index]


# ------------------------------------------------------------------- preparing

def heavy_atoms(pf: PDBFrame) -> PDBFrame:
    """Drop hydrogens by name: atom names starting with ``H``."""
    return pf.select(mask=~pf.df['atom_name'].str.startswith('H').to_numpy())


def is_annotated(pf: PDBFrame) -> bool:
    """Whether `pf` already has the standard chain IDs and IMGT labels (an ``imgt`` column)."""
    return 'imgt' in pf.df


def annotate(structure, name: str | None = None, complex: int | None = 1) -> TCRFrame:
    """A structure with the standard chain IDs and IMGT labels, as a
    :class:`~tcrstruct.TCRFrame`; all atoms kept.

    Detects the chains (:func:`detect_chains`), renames them to the standard IDs
    (A-F) in alphabetical order and adds two columns: ``imgt``, each residue's
    IMGT label (TCR chains: :func:`.numbering.imgt_labels`; MHC chains A / B: the
    G-domain position, :func:`tcrkit.annotate_mhc`; ``'-'`` elsewhere), and
    ``original_chain``, its chain ID in the input. Residue numbers stay as in
    the file. HETATM records (ligands, water, modified residues) are not read.
    A structure that is already annotated is only made a TCRFrame (a TCRFrame
    is returned as it is).

    Args:
        complex: for a file with several copies of the complex (two MHC groove
            chains, two TCR alphas, ...), which one to keep, by contacts (from 1;
            see :func:`.complexes.complex_table`), with a warning naming the chains
            left out; ``metadata['complex']``, ``['complex_chains']`` and
            ``['n_complexes']`` record it. ``None`` keeps every chain, the first of
            each kind by chain ID getting its standard ID.
    """
    from .tcr import TCRFrame
    pf = PDBFrame.ensure(structure)
    if isinstance(pf, TCRFrame) and name is None:
        return pf
    metadata = pf.metadata
    if not is_annotated(pf):
        pf = _annotate(pf, name, complex)[0]
        metadata = pf.metadata
    return TCRFrame._wrap(pf.df, name or pf.name, metadata, pf._seqres)


def _annotate(pf: PDBFrame, name: str | None = None, complex: int | None = None
              ) -> tuple[PDBFrame, pd.DataFrame]:
    """:func:`annotate`, also returning the chain table of :func:`detect_chains`."""
    seqs = dict(sorted(pf.sequences.items()))
    full = _full_sequences(pf, seqs, name or pf.name)
    numbering = run_anarci({c: full[c][0] if c in full else s for c, s in seqs.items()})
    table = detect_chains(pf, numbering)
    metadata = pf.metadata
    if complex is not None:
        from .complexes import _pick_complex
        part, picked = _pick_complex(pf, table, complex, name or pf.name)
        if picked:
            metadata = {**(metadata or {}), **picked}
            pf = part
            seqs = dict(sorted(pf.sequences.items()))
            numbering = {c: n for c, n in numbering.items() if c in seqs}
            table = detect_chains(pf, numbering)

    def on_residues(c, labels):
        """Labels of chain c's sequence (deposited, if read) on its residues with atoms."""
        return [labels[k] for k in full[c][1]] if c in full else list(labels)

    numbered = {c: full[c][0] if c in full else seqs[c] for c in seqs}
    every = {c: list(imgt_labels(len(numbered[c]), numbering.get(c))) for c in seqs}
    mhc = {'A', 'B'} if 'B' in set(table['new_chain']) else {'A'}
    for c, new in table['new_chain'].items():
        if new not in mhc:
            continue
        num = mhc_annotation(numbered[c])        # tcrkit's one call for this chain
        every[c] = list(num['imgt'])
        missing = [d for d in gene_domains(num['gene'].iloc[0])
                   if d not in set(num['domain'])]
        if missing:
            warn(f"{name or pf.name}: chain {new} (file chain {c}): no IMGT {', '.join(missing)} "
                 "domain found; its residues get imgt '-'")
    labels = {c: on_residues(c, every[c]) for c in seqs}
    df = pf.df.copy()
    # observed_index is the residue's 1-based position in its chain's sequence
    df['imgt'] = [labels[c][i - 1] for c, i in zip(df['chain'], df['observed_index'])]
    df['original_chain'] = df['chain']
    df['chain'] = df['chain'].map(table['new_chain'])
    # chains in alphabetical order, residues and atoms keep their order
    df = df.iloc[np.argsort(df['chain'].to_numpy(), kind='stable')]
    out = PDBFrame(df, name=name or pf.name, metadata=metadata)
    out._seqres = _labelled_seqres(pf, full, every, table['new_chain'])
    return out, table


def _labelled_seqres(pf: PDBFrame, full: dict, every: dict[str, list[str]],
                     new_chain: pd.Series) -> pd.DataFrame | None:
    """`pf`'s deposited sequence (PDBFrame's, behind ``pf.residues``) on the standard
    chain IDs, with each residue's ``imgt`` label, so unobserved residues have one too."""
    seqres = pf._seqres
    if seqres is None:
        return None
    res = PDBFrame.residues.fget(pf)
    imgt = {}
    for c, labels in every.items():
        rows = res[res['chain'] == c]
        if c not in full:                    # numbered on its residues with atoms
            rows = rows[rows['observed']]
        if len(rows) != len(labels):
            continue
        imgt.update({(c, int(i)): lab for i, lab in zip(rows['full_index'], labels)
                     if pd.notna(i)})
    out = seqres.assign(imgt=[imgt.get((c, int(i)), '-') for c, i in
                              zip(seqres['chain'], seqres['full_index'])])
    out['chain'] = out['chain'].map(new_chain).fillna(out['chain'])
    return out


def _full_sequences(pf: PDBFrame, seqs: dict[str, str], name: str
                    ) -> dict[str, tuple[str, list[int]]]:
    """:func:`.numbering.full_sequence` of every chain that has one, with a warning
    for a chain whose residues do not fit it (that one is numbered as it is)."""
    out = {}
    for c in seqs:
        try:
            got = full_sequence(pf, c)
        except ValueError as e:
            warn(f'{name}: chain {c}: {e} (_pdbx_poly_seq_scheme), so it is numbered from '
                 'its residues with coordinates only; IMGT labels after a gap may be shifted')
            continue
        if got is not None:
            out[c] = got
    return out


def scoring_frame(structure) -> PDBFrame:
    """What the metrics are computed on: the annotated structure, heavy atoms only."""
    pf = annotate(structure)
    return PDBFrame(heavy_atoms(pf).df, name=pf.name)


def normalize(structure, keep_leader: bool = False, heavy: bool = False,
              name: str | None = None, complex: int | None = 1) -> TCRFrame:
    """A TCR-pMHC in the standard form, ready to write to a file.

    Chains get the standard IDs (A-F, in that order) and the TCR chains (D, E)
    are renumbered by IMGT: residue 111A becomes number 111, insertion code A;
    residues past the V domain continue from its last position (129, 130, ...).
    Other chains keep their numbering.

    Args:
        keep_leader: keep TCR residues before the V domain, numbered -k ... -1.
            By default they are dropped.
        heavy: drop hydrogens too.
        complex: for a file with several copies of the complex, which one to keep
            (see :func:`annotate`).
    """
    pf = annotate(structure, complex=complex)
    seqres = pf._seqres
    if not keep_leader:
        leader = _imgt_numbered(pf.df) & pf.df['imgt'].str.startswith('-').to_numpy()
        unseen = None if seqres is None else ~(
            _imgt_numbered(seqres) & seqres['imgt'].str.startswith('-').to_numpy())
        pf = PDBFrame._subset(pf, ~leader, unseen)
        seqres = pf._seqres
    df = pf.df.copy()
    tcr = _imgt_numbered(df)
    parts = df.loc[tcr, 'imgt'].str.extract(r'^(-?\d+)(\D?)$')
    df.loc[tcr, 'residue_number'] = parts[0].astype(int)
    df.loc[tcr, 'insertion_code'] = parts[1]
    frame = PDBFrame(df, name=name or pf.name, metadata=pf.metadata)
    if seqres is not None:
        # residues with atoms as in df; unobserved TCR residues by their IMGT labels too
        seqres = deposited.remap(seqres, pf.df, df)
        unseen = _imgt_numbered(seqres) & ~seqres['observed'].to_numpy()
        parts = seqres.loc[unseen, 'imgt'].str.extract(r'^(-?\d+)(\D?)$')
        seqres.loc[unseen, 'residue_number'] = parts[0].astype(int)
        seqres.loc[unseen, 'insertion_code'] = parts[1]
        frame._seqres = seqres
    out = annotate(frame)
    return heavy_atoms(out) if heavy else out


def _imgt_numbered(table: pd.DataFrame) -> np.ndarray:
    """Rows of a TCR chain (D, E) with an IMGT label (atoms or deposited residues)."""
    if 'imgt' not in table:
        return np.zeros(len(table), bool)
    return (table['chain'].isin(['D', 'E']) & (table['imgt'] != '-')).to_numpy()
