"""Score a predicted TCR-pMHC structure against its reference: one call, one row.

    >>> from tcrstruct import compare
    >>> compare('pred/1AO7.cif', 'rcsb/1AO7.pdb')      # a pandas Series
    >>> compare('pred/1AO7.cif', 'rcsb/1AO7.pdb', **V1)  # the 'v1' metrics
    >>> pd.DataFrame([compare(m, r) for m, r in pairs])

The row has the columns of the ``'v1'`` metrics table (see :data:`COLUMNS`), then :data:`NEW_COLUMNS` and the superposition it
was computed in (:data:`SETTINGS`). :data:`METRIC_GROUPS` sorts the columns by what
they measure, and ``grouped=True`` returns them so, as ``(group, column)``;
``docs/metrics.md`` explains each. Two columns are named differently from that
table (:data:`REFERENCE_NAMES`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pdbframe import PDBFrame

from . import metrics
from .angles import ANGLE_METHODS, ANGLE_PARTS, angle_settings
from ._warnings import warn
from .chains import scoring_frame
from .regions import CDRS

__all__ = ['compare', 'read_processed', 'versions', 'group_columns', 'COLUMNS', 'NEW_COLUMNS',
           'SETTINGS', 'METRIC_GROUPS', 'REFERENCE_NAMES', 'V1']

#: The settings of the ``'v1'`` metrics:
#: ``compare(model, reference, **V1)``. See ``docs/metrics.md``.
V1 = {'align_atoms': 'heavy', 'angle_method': 'v1', 'dockq_method': 'v1', 'tm_method': 'v1'}

#: Our names of the ``'v1'`` table's columns, where they differ: ours -> the table's.
REFERENCE_NAMES = {'Full': 'Full_RMSD', 'Peptide': 'Peptide_RMSD'}

#: Output columns, in the ``'v1'`` table's order (two renamed, :data:`REFERENCE_NAMES`).
COLUMNS = (['PDB', 'Full', 'Peptide', 'Peptide_Centric', *CDRS, 'Mean_CDR',
            'crossing_angle', 'incident_angle', 'crossing_angle_error',
            'incident_angle_error', 'DockQ', 'iRMSD', 'LRMSD', 'fnat', 'nat_correct',
            'nat_total', 'fnonnat', 'nonnat_count', 'model_total', 'F1', 'clashes',
            'len1', 'len2', 'lddt', 'tm_score', 'alignment_identity'])

#: Columns the ``'v1'`` table does not have, after :data:`COLUMNS`: the Cα RMSD of
#: the TCR and of the pMHC in the superposition (:func:`.metrics.docking_rmsds`), and
#: whether the model's peptide runs backwards (:func:`.metrics.peptide_reversed`).
NEW_COLUMNS = ['TCR_LRMSD', 'pMHC_LRMSD', 'peptide_reversed']

#: What every row records last: the superposition it was scored in, as
#: :func:`compare`'s ``align_on`` and ``align_atoms``. The package versions are in
#: ``row.attrs['versions']`` (see :func:`versions`).
SETTINGS = ['align_on', 'align_atoms']

#: Every column by what it measures, in this order: ``rows[METRIC_GROUPS['docking']]``,
#: or ``compare(..., grouped=True)`` for ``(group, column)`` columns. RMSD columns are
#: named by their region (``Full``, ``Peptide``, ``CDRA1``, ...).
METRIC_GROUPS = {
    'pair': ['PDB', 'model', 'method', 'model_file', 'reference_file'],
    'global': ['Full', 'lddt', 'tm_score'],
    'peptide': ['Peptide', 'Peptide_Centric', 'peptide_reversed'],
    'cdrs': [*CDRS, 'Mean_CDR'],
    'docking': ['DockQ', 'iRMSD', 'LRMSD', 'TCR_LRMSD', 'pMHC_LRMSD', 'fnat', 'F1'],
    'angles': ['crossing_angle', 'incident_angle', 'crossing_angle_error',
               'incident_angle_error'],
    'contacts': ['nat_correct', 'nat_total', 'fnonnat', 'nonnat_count', 'model_total',
                 'clashes'],
    'bookkeeping': ['len1', 'len2', 'alignment_identity', *SETTINGS],
    'status': ['error', 'warnings'],
}


def versions() -> str:
    """The tcrstruct, PDBFrame and tcrkit versions, as recorded with every score."""
    import pdbframe
    import tcrkit

    from . import __version__
    return (f'tcrstruct {__version__}, pdbframe {pdbframe.__version__}, '
            f'tcrkit {tcrkit.__version__}')


def _settings(align_on='peptide_centric', align_atoms: str = 'ca', **kwargs) -> dict:
    """The :data:`SETTINGS` values of a :func:`compare` call with these arguments (one
    superposition)."""
    return {'align_on': align_on, 'align_atoms': align_atoms}


def _modes(align_on) -> list[str]:
    """`align_on` as a list of superposition modes (a list for several)."""
    modes = [align_on] if isinstance(align_on, str) else list(align_on)
    bad = [m for m in modes if m not in metrics.SUPERPOSITION_MODES]
    if bad or not modes:
        raise ValueError(f'align_on must be one of {list(metrics.SUPERPOSITION_MODES)} '
                         f'or a list of them, got {align_on!r}')
    return list(dict.fromkeys(modes))


def group_columns(table):
    """A :func:`compare` row (Series) or table (DataFrame) with its columns as
    ``(group, column)`` in :data:`METRIC_GROUPS` order; a new object."""
    is_row = isinstance(table, pd.Series)
    names = list(table.index if is_row else table.columns)
    group_of = {c: g for g, cols in METRIC_GROUPS.items() for c in cols}
    rank = {g: i for i, g in enumerate(METRIC_GROUPS)}
    order = sorted(names, key=lambda c: (rank.get(group_of.get(c), len(rank)), names.index(c)))
    index = pd.MultiIndex.from_tuples([(group_of.get(c, 'other'), c) for c in order],
                                      names=['group', 'column'])
    out = table.reindex(order) if is_row else table.reindex(columns=order)
    if is_row:
        out.index = index
    else:
        out.columns = index
    out.attrs = dict(table.attrs)
    return out


def compare(model, reference, min_identity: float = metrics.MIN_IDENTITY,
            align_on: str | list[str] = 'peptide_centric', name: str | None = None,
            angle_method: str | dict = 'piercelab', dockq_method: str = 'v2',
            tm_method: str = 'v2', align_atoms: str = 'ca',
            grouped: bool = False) -> pd.Series | pd.DataFrame:
    """All benchmark metrics of `model` against `reference`.

    Args:
        model, reference: structure files or PDBFrames. TCRFrames (and frames from
            :func:`read_processed`) are used as they are; plain ones are annotated
            first (chain detection, ANARCI).
        min_identity: chains less identical than this to the reference are not
            scored (see :func:`.metrics.match`).
        align_on: what the one superposition fits on - ``'peptide_centric'``
            (default), ``'full'``, ``'pmhc'`` (chains A, B, C,
            F), ``'mhc'`` (A, B), ``'tcr'`` (D, E), ``'v_domains'`` (the TCR V
            domains) or ``'cdrs'`` (the six CDRs); see
            :data:`.metrics.SUPERPOSITION_MODES`. Every RMSD is measured in that pose.
            A list scores several superpositions at once: a DataFrame with one row
            per superposition, the metrics that need no superposition computed once.
        align_atoms: the atoms of that region the superposition is fitted on:
            ``'ca'`` (default) or ``'heavy'`` (``'v1'``). RMSDs are
            of heavy atoms either way (the ``*_LRMSD`` of Cα atoms).
        name: the ``PDB`` value; defaults to the first 4 letters of the
            reference's name, upper-cased.
        angle_method: how docking angles are computed - ``'piercelab'`` (default:
            the Pierce tool's helix residues, V-domain rotation axis, 0-180
            degrees) or ``'v1'`` (Motif Atlas helices,
            whole-chain axis, folded to 0-90), or a dict mixing the two part by
            part, e.g. ``{'helices': 'v1'}``; see :mod:`.angles`.
        dockq_method: ``'v2'`` (default, DockQ v2: TCR against pMHC, its own
            superposition) or ``'v1'`` (peptide core
            against MHC and CDRs, in the `align_on` pose); see :func:`.metrics.dockq`.
        tm_method: ``'v2'`` (default: on its own fit of all matched Cα atoms) or
            ``'v1'`` (in the `align_on` pose); both normalised by the matched
            residues; see :func:`.metrics.tm_score`.
        grouped: give the columns as ``(group, column)`` (:func:`group_columns`), e.g.
            ``row['docking']``.

    :data:`V1` sets the fit and the three methods to ``'v1'``. The
    row has :data:`COLUMNS`, then :data:`NEW_COLUMNS`, then the superposition it was
    scored in (:data:`SETTINGS`); ``row.attrs['versions']`` has the package versions.
    A metric that fails leaves its columns empty (NaN) with a
    :class:`TCRStructWarning`, and docking angles and DockQ are only computed when
    the model has both TCR chains. Metrics use heavy atoms only.
    """
    try:
        angle_method = angle_settings(angle_method)
    except (KeyError, ValueError) as e:
        raise ValueError(f'angle_method must be one of {ANGLE_METHODS} or a dict of '
                         f'{list(ANGLE_PARTS)}: {e}') from None
    for arg, value, allowed in [('dockq_method', dockq_method, metrics.METRIC_METHODS),
                                ('tm_method', tm_method, metrics.METRIC_METHODS),
                                ('align_atoms', align_atoms, metrics.FIT_ATOMS)]:
        if value not in allowed:
            raise ValueError(f'{arg} must be one of {allowed}, got {value!r}')
    modes = _modes(align_on)
    ref_full, mod_full = scoring_frame(reference), scoring_frame(model)
    ref, matched, identity = metrics.match(ref_full, mod_full, min_identity=min_identity)

    tcr = {'D', 'E'} <= set(matched.chains)
    if not tcr:
        warn(f'{matched.name}: no TCR alpha and beta chains (D, E) matched; '
             'docking angles and DockQ not computed')
    # (label, function of (reference, model), run it, needs the superposition)
    steps = [('rmsd', metrics.rmsds, True, True),
             ('angles', lambda r, m: metrics.angles(r, m, method=angle_method), tcr, False),
             # DockQ v2 pairs residues and superposes by itself, on the whole structures
             ('dockq', lambda r, m: metrics.dockq(r, m, method='v1') if dockq_method == 'v1'
              else metrics.dockq(ref_full, mod_full, method='v2'), tcr, dockq_method == 'v1'),
             ('lddt', metrics.lddt, True, False),
             ('tm_score', lambda r, m: metrics.tm_score(r, m, method=tm_method), True,
              tm_method == 'v1'),
             ('docking RMSDs', metrics.docking_rmsds, True, True),
             ('peptide_reversed', metrics.peptide_reversed, 'C' in matched.chains, False)]

    def run(mod, posed: bool, where: str = '') -> dict:
        row = {}
        for label, fn, wanted, needs_pose in steps:
            if not wanted or needs_pose != posed:
                continue
            try:
                value = fn(ref, mod)
            except Exception as e:
                warn(f'{mod.name}: {label} not computed{where}: {e}')
                continue
            row.update(value if isinstance(value, dict) else {label: value})
        return row

    shared = {'PDB': name or ref.name[:4].upper(), **run(matched, posed=False),
              'alignment_identity': min(identity.values())}
    rows = []
    for mode in modes:
        mod = metrics.superpose(ref, matched, mode=mode, atoms=align_atoms)
        where = f' (align_on={mode!r})' if len(modes) > 1 else ''
        row = {**shared, **run(mod, posed=True, where=where),
               **_settings(align_on=mode, align_atoms=align_atoms)}
        rows.append(pd.Series(row, name=mod.name).reindex(COLUMNS + NEW_COLUMNS + SETTINGS))
    if isinstance(align_on, str):
        out = rows[0]
    else:
        out = pd.DataFrame(rows, index=modes)    # also the align_on column
    out.attrs['versions'] = versions()
    return group_columns(out) if grouped else out


# ----------------------------------------------------------- prepared structures

def read_processed(path: str | Path) -> PDBFrame:
    """A PDBFrame from a ``*_processed.csv`` file: a prepared structure.

    Those files hold a structure after preparation: standard chain IDs, heavy atoms
    and an ``imgt_index`` column (here: ``imgt``). The expected ``'v1'`` values in
    ``tests/data`` were scored from them, so reading them reproduces those inputs.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype={'residue_index': str, 'imgt_index': str},
                     keep_default_na=False)
    parts = df['residue_index'].str.extract(r'^(-?\d+)(\D*)$')
    if parts[0].isna().any():
        bad = df.loc[parts[0].isna(), 'residue_index'].iloc[0]
        raise ValueError(f'{path.name}: cannot read residue number {bad!r}')
    out = pd.DataFrame({
        'chain': df['chain'], 'residue_number': parts[0].astype(int),
        'insertion_code': parts[1], 'residue_name': df['residue_name'],
        'atom_name': df['atom_name'], 'element': df['element'],
        'altloc': df['altloc'].str.strip(), 'x': df['x'], 'y': df['y'], 'z': df['z'],
        'occupancy': df['occupancy'], 'bfactor': df['bfactor'],
        'imgt': df['imgt_index'].replace('', '-'),
    })
    # residues are ordered by residue_index_int; keep that order
    out = out.iloc[np.lexsort((np.arange(len(df)), df['residue_index_int'].to_numpy(),
                               df['chain'].to_numpy()))]
    return PDBFrame(out, name=path.name.split('.')[0].removesuffix('_processed'))
