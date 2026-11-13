"""A TCR-pMHC split into its parts - TCR, pMHC, MHC, peptide, or any region - as
frames or files.

    >>> parts = split(s)                                  # {'tcr': TCRFrame, 'pmhc': ...}
    >>> split(s, out_dir='parts/')                        # 1AO7_tcr.pdb, 1AO7_pmhc.pdb, ...
    >>> split(s, parts=['cdrs', 'peptide_core'], out_dir='parts/', fmt='cif')

Every part keeps the standard chain IDs and the file's residue numbers (use
:func:`.chains.normalize` first for IMGT numbers in the files).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._warnings import warn

if TYPE_CHECKING:
    from .tcr import TCRFrame

__all__ = ['split', 'SPLIT_PARTS']

#: The default parts and the regions each is made of. Any other region or group name
#: (``'cdrs'``, ``'cdr3'``, ``'peptide_core'``, ``'Va'``, ...) can be asked for too.
SPLIT_PARTS = {
    'tcr': ['TCRa', 'TCRb'],
    'pmhc': ['MHC', 'B2M', 'peptide'],
    'mhc': ['MHC', 'B2M'],
    'peptide': ['peptide'],
}


def split(structure, parts=tuple(SPLIT_PARTS), out_dir: str | Path | None = None,
          fmt: str = 'pdb') -> dict[str, TCRFrame] | dict[str, Path]:
    """The parts of a TCR-pMHC (or pMHC), each as its own structure.

    Args:
        structure: a file, PDBFrame or TCRFrame.
        parts: part names, from :data:`SPLIT_PARTS` (default: all four) or any
            region or group (see ``tcs.list_regions()``).
        out_dir: write each part to ``<out_dir>/<name>_<part>.<fmt>`` and return
            the paths instead of the frames.
        fmt: ``'pdb'`` (default) or ``'cif'``, for `out_dir`.

    Returns:
        ``{part: TCRFrame}``, or ``{part: path}`` with `out_dir`. A part the
        structure lacks (no TCR in a pMHC, ...) is left out with a warning.
    """
    from .tcr import TCRFrame, expand_regions
    pf = TCRFrame.ensure(structure)
    if isinstance(parts, str):
        parts = [parts]
    if fmt not in ('pdb', 'cif'):
        raise ValueError(f"fmt must be 'pdb' or 'cif', got {fmt!r}")
    out = {}
    for part in parts:
        regions = SPLIT_PARTS.get(part) or expand_regions(part)
        present = [r for r in regions if r in pf.regions]
        if not present:
            warn(f'{pf.name}: no {part} ({", ".join(regions)}) in this structure; part '
                 'left out', stacklevel=3)
            continue
        out[part] = pf.select(regions=present)
    if out_dir is None:
        return out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {part: frame.write(out_dir / f'{pf.name}_{part}.{fmt}') for part, frame in out.items()}
