"""IMGT numbering of TCR chains with ANARCI, and the IMGT label of every residue.

ANARCI comes from `tcrkit <https://github.com/akdslab/tcrkit>`_, a dependency
installed in this project's environment. tcrkit runs it in-process with a pyhmmer
backend, so no ``hmmscan`` binary or conda env is needed. It runs with the IMGT
scheme, germline assignment and every chain type allowed.

    >>> numbers = run_anarci({'D': 'KEVEQNSGPLSVPEG...', 'E': '...'})
    >>> numbers['D'].chain_type, numbers['D'].labels[:3], numbers['D'].v_gene
    ('A', ['2', '3', '4'], 'TRAV12-2*01')
    >>> imgt_labels(len(seq), numbers['D'])     # one label per residue

Labels are strings such as ``'27'`` or ``'111A'``. Residues after the numbered V
domain continue from its last position (``'129'``, ``'130'``, ...), residues before
it count up to it from below (``'-3'``, ``'-2'``, ``'-1'``), and chains ANARCI does
not number get ``'-'`` - as in the ``imgt_index`` column of :func:`.compare.read_processed`'s files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pdbframe import PDBFrame

__all__ = ['Numbering', 'run_anarci', 'imgt_labels', 'imgt_position', 'full_sequence',
           'CDR_IMGT']

#: IMGT CDR ranges, inclusive.
CDR_IMGT = {1: (27, 38), 2: (56, 65), 3: (105, 117)}


@dataclass(frozen=True)
class Numbering:
    """ANARCI's IMGT numbering of one chain's first variable domain.

    Attributes:
        labels: IMGT label of each numbered residue, in sequence order.
        residues: the numbered residues, one letter each (same length as `labels`).
        query_start: 0-based position in the chain where the domain starts.
        chain_type: ANARCI's chain type: ``'A'`` / ``'B'`` (TCR alpha / beta),
            ``'D'``, ``'G'``, or ``'H'``, ``'K'``, ``'L'`` for antibodies.
        v_gene, j_gene: the closest germline genes, e.g. ``'TRAV12-2*01'``.
        v_identity, j_identity: ANARCI's sequence identity to them (0-1; NaN if none).
        species: the species of those germline genes, ``'human'`` or ``'mouse'`` (the
            TCR germlines ANARCI has); ``'-'`` if none.
    """
    labels: list[str]
    residues: str
    query_start: int
    chain_type: str
    v_gene: str = '-'
    j_gene: str = '-'
    v_identity: float = float('nan')
    j_identity: float = float('nan')
    species: str = '-'


def run_anarci(sequences: dict[str, str]) -> dict[str, Numbering | None]:
    """Number each sequence with ANARCI (IMGT scheme, germline assignment on).

    Args:
        sequences: ``{chain: one-letter sequence}``.

    Returns:
        ``{chain: Numbering}``, with ``None`` for chains ANARCI does not number
        (MHC, B2M, peptide, ...).
    """
    try:
        import tcrkit
    except ImportError as e:
        raise ImportError('ANARCI comes from tcrkit: run `uv sync` in this project, or '
                          'pip install git+ssh://git@github.com/akdslab/tcrkit.git') from e
    if not sequences:
        return {}
    table = tcrkit.anarci(dict(sequences), include_imgt_index=True, progress_bar=False)
    out = {}
    for chain, row in table.iterrows():
        numbered = row.get('imgt_dict')          # no column when no chain is numbered
        numbered = numbered if isinstance(numbered, dict) else {}
        if row['error'] != '-' or not numbered:
            out[chain] = None
            continue
        out[chain] = Numbering(labels=list(numbered), residues=''.join(numbered.values()),
                               query_start=int(row['query_start']),
                               chain_type=row['chain_type'],
                               v_gene=row['v_gene'], j_gene=row['j_gene'],
                               v_identity=_number(row.get('v_identity')),
                               j_identity=_number(row.get('j_identity')),
                               species=_text(row.get('germline_species')))
    return out


def _number(value) -> float:
    """`value` as a float, NaN for ANARCI's ``'-'`` or a missing value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _text(value) -> str:
    """`value` as a string, ``'-'`` for a missing value."""
    return value if isinstance(value, str) and value else '-'


def imgt_labels(n_residues: int, numbering: Numbering | None) -> list[str]:
    """IMGT label of every residue of a chain of `n_residues` residues.

    The numbered domain covers the residues from ``numbering.query_start`` on;
    residues past it continue from the last IMGT position, residues before it get
    ``'-k' ... '-1'``. Without a numbering, every residue is ``'-'``.
    """
    if numbering is None:
        return ['-'] * n_residues
    start, labels = numbering.query_start, numbering.labels
    tail = n_residues - start - len(labels)
    if tail < 0:
        raise ValueError(f'numbering covers {start + len(labels)} residues, chain has '
                         f'{n_residues}')
    last = imgt_position(labels[-1])
    return ([str(i) for i in range(-start, 0)] + list(labels)
            + [str(last + i) for i in range(1, tail + 1)])


def imgt_position(label: str) -> int | None:
    """The leading integer of an IMGT label (``'111A'`` -> 111); None for ``'-'``."""
    m = re.match(r'^-?\d+', label)
    return int(m.group()) if m else None


def full_sequence(pf, chain: str) -> tuple[str, list[int]] | None:
    """A chain's deposited sequence, and where each of its residues with atoms sits in it.

    A structure file has atoms only for the residues that were observed. Numbering only
    those runs a gap's two sides together, so ANARCI shifts the IMGT labels around it.
    Read from mmCIF, ``pf.residues`` also has the unobserved residues (PDBFrame's
    ``observed`` column); this is what gets numbered when there are any, and each label
    then goes on its residue.

    Returns:
        ``(sequence, positions)``: the chain's residues with the unobserved ones, and
        for each residue with atoms (in order) its 0-based position there. None when
        the chain has no unobserved residues.

    Raises:
        ValueError: the chain's residues do not fit its deposited sequence (a residue
            with atoms missing from it, or out of order).
    """
    res = PDBFrame.residues.fget(pf)        # the plain table, also for a TCRFrame
    rows = res[res['chain'] == chain].reset_index(drop=True)
    if rows.empty or rows['observed'].all():
        return None
    seen = rows[rows['observed']]
    outside = seen[seen['full_index'].isna()]
    if len(outside):
        n, i, aa = outside.iloc[0][['residue_number', 'insertion_code', 'aa']]
        raise ValueError(f'residue {n}{i} {aa} is not in its deposited sequence')
    if not seen['full_index'].is_monotonic_increasing:
        raise ValueError('its residues are out of order in its deposited sequence')
    return ''.join(rows['aa']), seen.index.tolist()
