"""TCR docking angles: crossing and incident angle over the MHC groove.

    >>> docking_angles(pf)                        # {'crossing_angle': ..., 'incident_angle': ...}
    >>> docking_angles(pf, method='v1')           # the 'v1' angles
    >>> docking_angles(pf, helices='v1')          # Pierce lab's, except for the helix residues
    >>> docking_vectors(pf)                       # the vectors behind them, for drawing
    >>> pf.transform(*tcr_up_orientation(pf))   # TCR up, peptide left to right

- **crossing angle**: between the long axis of the MHC groove helices and the
  line joining the TCR alpha and beta disulfides (Cys IMGT 23 / 104).
- **incident angle**: between the normal of the groove plane and the TCR
  alpha/beta inter-domain rotation axis.

Groove axes come from an SVD of the helix Cα atoms (:func:`.mhc.mhc_helices`).
Two methods (:data:`ANGLE_METHODS`), each a choice for three parts
(:data:`ANGLE_PARTS`) that can also be set one by one:

=========== ================================== ===================================
part        ``'piercelab'`` (default)          ``'v1'``
=========== ================================== ===================================
``helices`` ``'imgt'``: fixed IMGT positions   ``'v1'``: the Motif Atlas
            (the Pierce tool's residues,       gene's helix ranges
            :data:`.mhc.PIERCE_HELICES`)
``axis``    rotation axis from the V domains   ``'v1'``: from every IMGT label both TCR
            only (IMGT 1-128)                  chains share, constant domains too
``signed``  ``True``: angles 0-180             ``False``: folded into 0-90
=========== ================================== ===================================

``method='piercelab'`` follows the approach of the Pierce lab's ``tcr_docking_angle``
(https://github.com/piercelab/tcr_docking_angle, CC-BY-4.0); it is our own
implementation, not a copy.

Why they differ:

- ``helices``: the Motif Atlas (http://mhcmotifatlas.org) ranges are shifted gene
  by gene for class II (DR alpha 48-80, DQ beta 56-93, DP beta 58-95), which moves
  the class II crossing angle 1-7 deg. Pierce's residues sit at the same place in
  every gene's helix: they are fixed IMGT G-domain positions (``'imgt'``,
  :data:`tcrkit.gdomain.HELIX`), so files numbered differently, and any allele,
  still work.
- ``axis``: past the V domain, IMGT labels are only counters, so pairing them
  across alpha and beta pairs residues that do not correspond. That makes the
  reference incident angle barely correlate with Pierce's. The V-domain axis is
  within about 0.6 deg of Pierce's, which pairs residues by structure (FAST)
  rather than by IMGT label.
- ``signed``: folding into 0-90 makes a reverse-docked TCR (crossing about 130)
  look canonical (about 50). Signed angles can always be folded, but folded
  angles cannot be unfolded.

With ``signed=True`` every vector has a direction, so a reverse-docked TCR
(crossing angle above 90) stays distinguishable from a canonical one. The
directions are set from the structure, not from residue numbers: the groove axis
runs along the first helix of chain A (alpha1; the alpha chain's helix in class II)
from its N- to its C-terminus, the TCR vector from the alpha to the beta disulfide,
the plane normal points to the TCR, and the rotation axis points from the CDRs into
the V domains. Peptides bind in that same N- to C-terminal direction; the groove
axis is taken from the MHC so that a peptide modelled backwards does not flip it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure.geometry import kabsch

from .numbering import CDR_IMGT

__all__ = ['docking_angles', 'docking_vectors', 'angle_settings', 'tcr_vector',
           'groove_axes', 'tcr_rotation_axis', 'tcr_up_orientation', 'orientations',
           'peptide_direction', 'ANGLE_METHODS', 'ANGLE_PARTS']

#: Ways to compute the angles; see the module docstring.
ANGLE_METHODS = ('piercelab', 'v1')

#: The parts a method sets, and their value under each method.
ANGLE_PARTS = {'helices': {'piercelab': 'imgt', 'v1': 'v1'},
               'axis': {'piercelab': 'piercelab', 'v1': 'v1'},
               'signed': {'piercelab': True, 'v1': False}}

_IMGT = re.compile(r'(\d+)[A-Z]?')
_CDR_POSITIONS = {i for lo, hi in CDR_IMGT.values() for i in range(lo, hi + 1)}


def _check_method(method: str) -> None:
    if method not in ANGLE_METHODS:
        raise ValueError(f'method must be one of {ANGLE_METHODS}, got {method!r}')


def angle_settings(method: str | dict = 'piercelab', **parts) -> dict:
    """The full settings ``{'helices': ..., 'axis': ..., 'signed': ...}`` of a method.

    Args:
        method: ``'piercelab'`` or ``'v1'``, or a dict of parts, e.g.
            ``{'helices': 'v1'}``; parts it leaves out follow its ``'method'``
            entry (default ``'piercelab'``).
        parts: ``helices=`` (``'imgt'`` / ``'v1'``), ``axis=`` (``'piercelab'`` /
            ``'v1'``) or ``signed=`` (bool) to override; ``None`` keeps the method's.

    >>> angle_settings('piercelab', helices='v1')
    {'helices': 'v1', 'axis': 'piercelab', 'signed': True}
    """
    if isinstance(method, dict):
        parts = {**method, **{k: v for k, v in parts.items() if v is not None}}
        method = parts.pop('method', 'piercelab')
    _check_method(method)
    unknown = set(parts) - set(ANGLE_PARTS)
    if unknown:
        raise KeyError(f'unknown angle parts {sorted(unknown)}; use {list(ANGLE_PARTS)}')
    out = {k: v[method] for k, v in ANGLE_PARTS.items()}
    for k, v in parts.items():
        if v is None:
            continue
        allowed = (True, False) if k == 'signed' else tuple(ANGLE_PARTS[k].values())
        if v not in allowed or (k == 'signed' and not isinstance(v, bool)):
            raise ValueError(f'{k} must be one of {allowed}, got {v!r}')
        out[k] = v
    return out


def tcr_vector(pf: PDBFrame) -> np.ndarray:
    """From the beta (E) to the alpha (D) disulfide: centroids of SG of Cys 23 and 104."""
    df = pf.df
    tcr = df['chain'].isin(['D', 'E'])
    cys_res = tcr & df['imgt'].isin(['23', '104'])
    n_res = len(df[cys_res].drop_duplicates(['chain', 'observed_index']))
    if n_res != 4:
        raise ValueError(f'expected 4 residues at IMGT 23/104 on chains D and E, got {n_res}')
    sg = df[cys_res & (df['atom_name'] == 'SG')]
    if len(sg) != 4:
        raise ValueError(f'expected 4 SG atoms at IMGT 23/104, got {len(sg)}')
    centroid = sg.groupby('chain')[['x', 'y', 'z']].mean()
    return (centroid.loc['D'] - centroid.loc['E']).to_numpy(float)


_NO_HELICES = ('no MHC groove-helix residues (chain A, and B for class II): the groove '
               'orientation needs them')


def groove_axes(pf: PDBFrame, helices: np.ndarray | None = None) -> np.ndarray:
    """Principal axes of the groove helix Cα atoms, as rows: long axis first, plane normal last.

    `helices` is an atom mask; default: :func:`.mhc.mhc_helices` (``method='imgt'``).
    """
    from .mhc import mhc_helices
    helices = mhc_helices(pf) if helices is None else helices
    ca = pf.coords[helices & (pf.df['atom_name'] == 'CA').to_numpy()]
    if len(ca) < 3:
        raise ValueError(_NO_HELICES)
    _, _, vh = np.linalg.svd(ca - ca.mean(axis=0), full_matrices=False)
    return vh


def tcr_rotation_axis(pf: PDBFrame, method: str = 'piercelab',
                      directed: bool | None = None) -> np.ndarray:
    """Axis of the rotation that superposes TCR beta (E) onto alpha (D).

    Residues are paired by IMGT label (Cα atoms).

    Args:
        method: ``'piercelab'``: V-domain residues only (IMGT 1-128).
            ``'v1'``: every label both chains share, in each chain's own
            order (labels before and past the V domain count too).
        directed: point the axis from the CDRs into the V domains. Default: yes
            for ``'piercelab'``, the rotation's own sign for ``'v1'``.
    """
    _check_method(method)
    directed = method == 'piercelab' if directed is None else directed
    df = pf.df
    a, b = df[df['chain'] == 'E'], df[df['chain'] == 'D']
    if method == 'v1':
        common = set(a['imgt']) & set(b['imgt'])
        ca_a = a[a['imgt'].isin(common) & (a['atom_name'] == 'CA')][['x', 'y', 'z']].to_numpy(float)
        ca_b = b[b['imgt'].isin(common) & (b['atom_name'] == 'CA')][['x', 'y', 'z']].to_numpy(float)
        R, _ = kabsch(ca_a, ca_b)
        R = R.T  # 'v1' rotates row vectors, ``x @ R``
        phi = np.arccos((np.trace(R) - 1) / 2)
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(phi))
        v = -v / np.linalg.norm(v)
        return _toward_v_domain(v, a, b) if directed else v

    ca_a, ca_b = _v_domain_ca(a), _v_domain_ca(b)
    labels = ca_a.index.intersection(ca_b.index)
    if len(labels) < 3:
        raise ValueError(f'{pf.name}: only {len(labels)} V-domain residues (IMGT 1-128) '
                         'are shared by chains D and E; are the TCRs IMGT-numbered?')
    R, _ = kabsch(ca_a.loc[labels].to_numpy(float), ca_b.loc[labels].to_numpy(float))
    # the axis is R's eigenvector of eigenvalue 1: stable also near 180 degrees
    w, vecs = np.linalg.eig(R)
    v = _unit(np.real(vecs[:, np.argmin(np.abs(w - 1))]))
    return _toward_v_domain(v, a, b) if directed else v


def _toward_v_domain(v: np.ndarray, a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """`v`, or -`v`, pointing from the CDRs into the V domains of chains `a` and `b`."""
    tcr = pd.concat([_v_domain_ca(a), _v_domain_ca(b)])
    in_cdr = [_imgt_number(lab) in _CDR_POSITIONS for lab in tcr.index]
    toward_domain = tcr.mean().to_numpy() - tcr[in_cdr].mean().to_numpy()
    return v if np.dot(v, toward_domain) >= 0 else -v


def _imgt_number(label: str) -> int | None:
    m = _IMGT.fullmatch(str(label))
    return int(m.group(1)) if m else None


def _v_domain_ca(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Cα coordinates of one chain's V domain (IMGT 1-128), indexed by IMGT label."""
    ca = chain_df[chain_df['atom_name'] == 'CA']
    numbers = ca['imgt'].map(_imgt_number)
    ca = ca[numbers.between(1, 128)]
    return ca.drop_duplicates('imgt').set_index('imgt')[['x', 'y', 'z']]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def docking_vectors(pf: PDBFrame, method: str | dict = 'piercelab', *, helices: str | None = None,
                    axis: str | None = None, signed: bool | None = None) -> dict:
    """The vectors the docking angles are measured between, with points to draw them at.

    Args:
        method, helices, axis, signed: as for :func:`docking_angles`.

    Returns:
        ``{'groove_axis', 'groove_normal', 'tcr_vector', 'rotation_axis'}``: unit
        vectors; ``'groove_center'``, ``'tcr_center'``: the centroid of the helix
        Cα atoms and the midpoint of the two TCR disulfides; ``'helices'``: the
        helix atom mask; ``'settings'``: the parts used. With ``signed=True`` the
        vectors have the directions of the module docstring; otherwise the groove
        axes are the SVD's and the TCR vector runs from beta to alpha, as in the
        reference.
    """
    from .mhc import mhc_helices
    s = angle_settings(method, helices=helices, axis=axis, signed=signed)
    mask = mhc_helices(pf, method=s['helices'])
    if not (mask & (pf.df['atom_name'] == 'CA').to_numpy()).any():
        raise ValueError(f'{pf.name}: no MHC helix residues found (chains A / B)')
    vh = groove_axes(pf, mask)
    beta_to_alpha = tcr_vector(pf)
    rotation = tcr_rotation_axis(pf, s['axis'], directed=s['signed'])
    if s['signed']:
        long_axis, normal = _directed_groove_axes(pf, vh, mask)
        tcr = _unit(-beta_to_alpha)
    else:
        long_axis, normal, tcr = _unit(vh[0]), _unit(vh[2]), _unit(beta_to_alpha)
    df = pf.df
    sg = df[df['chain'].isin(['D', 'E']) & df['imgt'].isin(['23', '104']) & (df['atom_name'] == 'SG')]
    return {'groove_axis': long_axis, 'groove_normal': normal, 'tcr_vector': tcr,
            'rotation_axis': rotation,
            'groove_center': pf.coords[mask & (df['atom_name'] == 'CA').to_numpy()].mean(axis=0),
            'tcr_center': sg[['x', 'y', 'z']].to_numpy(float).mean(axis=0),
            'helices': mask, 'settings': s}


def docking_angles(pf: PDBFrame, method: str | dict = 'piercelab', *, helices: str | None = None,
                   axis: str | None = None, signed: bool | None = None) -> dict[str, float]:
    """Crossing and incident angle (degrees) of an annotated TCR-pMHC.

    Args:
        method: ``'piercelab'`` (default) or ``'v1'`` (Motif Atlas helices,
            whole-chain axis, 0-90), or a dict of parts (see :func:`angle_settings`).
        helices: override the method's helix residues: ``'imgt'`` or ``'v1'``.
        axis: override the method's rotation axis: ``'piercelab'`` (V domains) or
            ``'v1'`` (all shared IMGT labels).
        signed: override the method's signs: ``True`` (0-180) or ``False`` (0-90).

    Raises if a part is missing (no Cys 23/104, no MHC helices, ...); see
    :func:`.metrics.angles` for the NaN-on-failure version the benchmark uses.
    """
    v = docking_vectors(pf, method, helices=helices, axis=axis, signed=signed)
    crossing = np.dot(v['groove_axis'], v['tcr_vector'])
    incident = np.dot(v['groove_normal'], v['rotation_axis'])
    if not v['settings']['signed']:
        crossing, incident = abs(crossing), abs(incident)
    return {'crossing_angle': float(np.degrees(np.arccos(np.clip(crossing, -1.0, 1.0)))),
            'incident_angle': float(np.degrees(np.arccos(np.clip(incident, -1.0, 1.0))))}


#: Residues between two helix Cα atoms of chain A that start a new helix (alpha2).
_HELIX_GAP = 10


def _alpha1_direction(pf: PDBFrame, helices: np.ndarray) -> np.ndarray:
    """N- to C-terminal direction of chain A's first groove helix (alpha1, or the
    alpha chain's helix in class II): its last helix Cα minus its first."""
    df = pf.df
    ca = (helices & (df['atom_name'] == 'CA').to_numpy() & (df['chain'] == 'A').to_numpy())
    index = df.loc[ca, 'observed_index'].to_numpy()
    xyz = pf.coords[ca][np.argsort(index, kind='stable')]
    gaps = np.flatnonzero(np.diff(np.sort(index)) > _HELIX_GAP)
    end = gaps[0] + 1 if len(gaps) else len(xyz)
    if end < 2:
        raise ValueError(f'{pf.name}: chain A: need 2 helix Cα atoms to orient the groove axis')
    return xyz[end - 1] - xyz[0]


def peptide_direction(pf: PDBFrame) -> float:
    """Cosine between the peptide core (first to last Cα) and the MHC alpha1 helix
    (N to C). Peptides bind N to C along alpha1, so it is positive in a correct
    structure and negative for a peptide modelled backwards."""
    from .mhc import mhc_helices
    from .regions import peptide_core
    is_ca = (pf.df['atom_name'] == 'CA').to_numpy()
    pep = pf.coords[peptide_core(pf) & is_ca]
    if len(pep) < 2:
        raise ValueError(f'{pf.name}: chain C: need 2 peptide core Cα atoms for its direction')
    return float(np.dot(_unit(pep[-1] - pep[0]),
                        _unit(_alpha1_direction(pf, mhc_helices(pf)))))


def _directed_groove_axes(pf: PDBFrame, vh: np.ndarray,
                          helices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Groove long axis along the alpha1 helix (N to C), plane normal toward the TCR."""
    df, xyz = pf.df, pf.coords
    is_ca = (df['atom_name'] == 'CA').to_numpy()
    tcr = xyz[df['chain'].isin(['D', 'E']).to_numpy()]
    center = xyz[helices & is_ca].mean(axis=0)
    long_axis, normal = _unit(vh[0]), _unit(vh[2])
    if np.dot(_alpha1_direction(pf, helices), long_axis) < 0:
        long_axis = -long_axis
    if np.dot(tcr.mean(axis=0) - center, normal) < 0:
        normal = -normal
    return long_axis, normal


def tcr_up_orientation(pf: PDBFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation ``(R, t)`` that turn `pf` TCR up, for viewing.

    Built from the groove axes the default angles use (:func:`groove_axes` of the
    Pierce helix residues):

    - origin: the centre of the groove helices;
    - y (up): the groove-plane normal, pointing to the TCR (to the peptide if
      there is no TCR);
    - x (right): the groove's long axis, along the alpha1 helix from its N- to its
      C-terminus (the direction peptides bind in);
    - z (toward the viewer): x cross y.

    3D viewers look down -z with y up, so ``pf.transform(R, t)`` shows the TCR on
    top and the peptide running left to right across the groove.
    """
    from .mhc import mhc_helices
    df, xyz = pf.df, pf.coords
    helices = mhc_helices(pf)
    helix_ca = xyz[helices & (df['atom_name'] == 'CA').to_numpy()]
    if len(helix_ca) < 3:
        raise ValueError(_NO_HELICES)
    center = helix_ca.mean(axis=0)
    vh = groove_axes(pf, helices)
    long_axis, normal = vh[0], vh[2]

    tcr = df['chain'].isin(['D', 'E']).to_numpy()
    pep = (df['chain'] == 'C').to_numpy()
    above = xyz[tcr] if tcr.any() else xyz[pep]
    if not len(above):
        raise ValueError('need a TCR (D, E) or a peptide (C) to tell up from down')
    if np.dot(above.mean(axis=0) - center, normal) < 0:
        normal = -normal
    if np.dot(_alpha1_direction(pf, helices), long_axis) < 0:
        long_axis = -long_axis
    x = long_axis - np.dot(long_axis, normal) * normal
    x /= np.linalg.norm(x)
    z = np.cross(x, normal)
    R = np.vstack([x, normal, z])
    return R, -center @ R.T


def orientations(pf: PDBFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named camera orientations for ``PDBFrame.view(orientations=...)``.

    ``'TCR up'`` is :func:`tcr_up_orientation`. ``'Groove from above'`` looks
    down onto the peptide from the TCR: the plane normal toward the viewer, the
    peptide still left to right.
    """
    R, t = tcr_up_orientation(pf)
    x, up = R[0], R[1]
    top = np.vstack([x, np.cross(up, x), up])
    return {'TCR up': (R, t), 'Groove from above': (top, -(-t @ R) @ top.T)}
