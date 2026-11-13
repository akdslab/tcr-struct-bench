"""Which residues touch which: contact maps between regions, as a table, a grid or a plot.

    >>> contact_map(s)                         # CDR residues within 5 Å of the peptide
    >>> contact_map(s, grid=True)              # the same as a CDR x peptide grid
    >>> plot_contact_map(s)                    # ... drawn as a heatmap
    >>> contact_map(s, rows='cdrs', columns='helices', cutoff=4.0)
    >>> cdr_contacts(s)                        # per CDR: how close to the peptide and MHC

Rows and columns are labelled ``'CDRB3 L110'`` (region, residue, IMGT position) or,
for the peptide, ``'Y5'`` (residue and position). Peptide positions count from the
peptide core: the whole peptide for class I, so ``L1 ... V9`` is the 9-mer
LLFGYPVYV; for class II, 1 is the core's first residue (P1), with the flanks at
``-1, -2, ...`` before it and ``10, 11, ...`` after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pdbframe.structure import deposited

__all__ = ['contact_map', 'plot_contact_map', 'cdr_contacts']

_KEYS = ['chain', 'residue_number', 'insertion_code']
_CDRS = ['CDRA1', 'CDRA2', 'CDRA3', 'CDRB1', 'CDRB2', 'CDRB3']


def cdr_contacts(pf, cutoff: float = 4.5) -> pd.DataFrame:
    """How each CDR sits on the pMHC: closest distances and touching residues.

    Args:
        pf: an annotated structure with a TCR (TCRFrame, file, ...).
        cutoff: the closest-heavy-atom distance (Å) that counts as touching.

    Returns:
        One row per CDR (the index): ``peptide_distance`` and ``mhc_distance`` (Å,
        closest heavy atoms; NaN if either side is missing), ``peptide_residues``
        and ``mhc_residues`` (the CDR's residues within `cutoff` of the peptide /
        the MHC chains A, B) and ``touches_pmhc``.
    """
    from scipy.spatial import cKDTree

    from .tcr import TCRFrame
    s = TCRFrame.ensure(pf)
    heavy = ~s.df['atom_name'].str.startswith('H').to_numpy()     # as chains.heavy_atoms
    xyz, chain = s.coords, s.df['chain'].to_numpy()
    sides = {'peptide': heavy & (chain == 'C'), 'mhc': heavy & np.isin(chain, ['A', 'B'])}
    trees = {k: cKDTree(xyz[m]) if m.any() else None for k, m in sides.items()}
    residue = (s.df['chain'] + ':' + s.df['observed_index'].astype(str)).to_numpy()
    rows = []
    for name in _CDRS:
        m = heavy & s.masks.get(name, np.zeros(len(s.df), bool))
        row = {'CDR': name}
        for side, tree in trees.items():
            if tree is None or not m.any():
                row[f'{side}_distance'], row[f'{side}_residues'] = np.nan, 0
                continue
            d = tree.query(xyz[m])[0]
            row[f'{side}_distance'] = float(d.min())
            row[f'{side}_residues'] = len(set(residue[m][d <= cutoff]))
        rows.append(row)
    out = pd.DataFrame(rows).set_index('CDR')
    out['touches_pmhc'] = (out['peptide_residues'] + out['mhc_residues']) > 0
    return out[['peptide_distance', 'mhc_distance', 'peptide_residues', 'mhc_residues',
                'touches_pmhc']]


def _labels(pf) -> pd.DataFrame:
    """Each residue of `pf` with its ``label`` (see the module docstring)."""
    res = pf.residues
    res = res[res['observed']].astype({'observed_index': int}).reset_index(drop=True)
    pep = (res['chain'] == 'C').to_numpy()
    if pep.any():
        core = pf.masks.get('peptide_core')
        index = res.loc[pep, 'observed_index'].to_numpy()
        start = (pf.df.loc[core, 'observed_index'].min() if core is not None and core.any()
                 else index.min())
        pos = index - start
        pos = np.where(pos >= 0, pos + 1, pos)       # no position 0: ..., -1, 1, 2, ...
        position = pd.Series(pos, index=res.index[pep]).astype(str)
        res.loc[pep, 'label'] = res.loc[pep, 'aa'] + position
    other = ~pep
    number = res['imgt'].where(res['imgt'] != '-', res['residue_number'].astype(str)
                               + res['insertion_code'].str.strip())
    res.loc[other, 'label'] = (res.loc[other, 'region'].fillna(res.loc[other, 'chain'])
                               + ' ' + res.loc[other, 'aa'] + number[other])
    return res.set_index(_KEYS)


def contact_map(pf, rows='cdrs', columns='peptide', cutoff: float = 5.0,
                grid: bool = False) -> pd.DataFrame:
    """Residue pairs of two regions within `cutoff` Å (closest heavy atoms).

    Args:
        pf: an annotated structure (TCRFrame, file, ...).
        rows, columns: regions or groups (as for ``TCRFrame.select(regions=)``), or
            lists of them. Default: the six CDRs against the peptide.
        cutoff: the largest closest-heavy-atom distance that counts as a contact.
        grid: return a grid instead of a table: one row per contacting `rows`
            residue (in sequence order), one column per `columns` residue (every
            one, in order), the distance in each cell, NaN for no contact.

    Returns:
        One row per contact: ``row``, ``column`` (the labels), ``distance``, and
        each side's ``chain``, ``region`` and ``imgt`` (``row_chain``, ...), closest
        first per row residue.
    """
    from .tcr import TCRFrame
    pf = TCRFrame.ensure(pf)
    a = pf.select(regions=rows, atoms='heavy')
    b = pf.select(regions=columns, atoms='heavy')
    labels = _labels(pf)
    found = a.contacts(b, cutoff=cutoff)
    out = {}
    for side, n in [('row', 1), ('column', 2)]:
        keys = list(zip(found[f'chain_{n}'], found[f'residue_number_{n}'],
                        found[f'insertion_code_{n}']))
        lab = labels.loc[keys] if keys else labels.iloc[:0]
        out[side] = lab['label'].to_numpy()
        out[f'{side}_chain'] = found[f'chain_{n}'].to_numpy()
        out[f'{side}_region'] = lab['region'].to_numpy()
        out[f'{side}_imgt'] = lab['imgt'].to_numpy()
        out[f'{side}_index'] = lab['observed_index'].to_numpy()
    table = pd.DataFrame({'row': out['row'], 'column': out['column'],
                          'distance': found['distance'].to_numpy(),
                          **{k: v for k, v in out.items() if k not in ('row', 'column')}})
    table = table.sort_values(['row_chain', 'row_index', 'distance'], kind='stable')
    if not grid:
        return table.drop(columns=['row_index', 'column_index']).reset_index(drop=True)
    order = labels.loc[list(map(tuple, deposited.observed_residues(b.df)[_KEYS].to_numpy()))]['label']
    out = table.pivot_table(index='row', columns='column', values='distance', aggfunc='min',
                            sort=False)
    return out.reindex(columns=list(order)).rename_axis(index=None, columns=None)


def plot_contact_map(pf, rows='cdrs', columns='peptide', cutoff: float = 5.0, ax=None):
    """:func:`contact_map` as a heatmap, darker for closer; empty cells are no contact.

    Args:
        pf: an annotated structure, or a grid from ``contact_map(..., grid=True)``.
        rows, columns, cutoff: as for :func:`contact_map`.
        ax: a matplotlib Axes to draw on (default: a new figure sized to the grid).

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(pf, pd.DataFrame):
        grid, name = pf, None
    else:
        from .tcr import TCRFrame
        pf = TCRFrame.ensure(pf)
        grid, name = contact_map(pf, rows, columns, cutoff, grid=True), pf.name
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.0, 0.4 * grid.shape[1] + 2),
                                      0.32 * len(grid) + 1))
    closer_darker = LinearSegmentedColormap.from_list(
        'closer_darker', ['#0d366b', '#256abf', '#6da7ec', '#cde2fb'])
    cells = ax.pcolormesh(grid.to_numpy(float), cmap=closer_darker, vmin=2.5, vmax=cutoff,
                          edgecolors='white', linewidth=2)
    ax.set_xticks([x + 0.5 for x in range(grid.shape[1])], grid.columns)
    ax.set_yticks([y + 0.5 for y in range(grid.shape[0])], grid.index)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.figure.colorbar(cells, ax=ax, shrink=0.6, label='closest heavy-atom distance (Å)')
    if name:
        ax.set_title(f'{name}: contacts within {cutoff:g} Å', loc='left', pad=24)
    return ax
