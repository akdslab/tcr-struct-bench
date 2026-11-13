"""The benchmark's scores of a predicted TCR-pMHC against its reference.

RMSDs and lDDT (see ``docs/metrics.md``): one
superposition per pair (peptide-centric by default) and no refit, so every RMSD is
measured in that single pose, on heavy atoms; lDDT over all heavy-atom pairs of
residues whose Cα atoms are within 15 Å.

The superposition is fitted on the Cα atoms of its region (``atoms='ca'``, default)
or on all its heavy atoms (``'heavy'``, ``'v1'``). DockQ and TM-score
come in two versions (``method=``); see ``docs/metrics.md``.

- ``'v2'`` (default), as the original programs:

  - DockQ, as DockQ v2 (https://github.com/wallnerlab/DockQ; Mirabello and Wallner,
    "DockQ v2: improved automatic quality measure for protein multimers, nucleic
    acids, and small molecules", *Bioinformatics* 40(10), btae586, 2024,
    https://doi.org/10.1093/bioinformatics/btae586), with the TCR (D, E) as one side
    and the pMHC (A, B, C, F) as the other. It superposes the structures itself.
  - TM-score of the whole complex on its own superposition: all matched Cα atoms
    fitted (Kabsch), normalised by the matched residues.

- ``'v1'``: DockQ of the peptide core against the MHC
  around it plus the CDRs, with no refit; TM-score in the current pose, normalised
  by the matched residues.

Two layers, as in PDBFrame: frame-level functions on a matched pair (from
:func:`match`), and ``*_coords`` functions on plain arrays.

    >>> ref, model, identity = match(ref, model)
    >>> model = superpose(ref, model)              # peptide-centric
    >>> rmsds(ref, model), dockq(ref, model), lddt(ref, model)
    >>> dockq(ref, model, method='v1'), tm_score(ref, model, method='v1')
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pdbframe import PDBFrame, SeqFrame
from pdbframe.structure import deposited
from pdbframe.structure.geometry import atom_pairs_within, kabsch, rmsd, transform
from pdbframe.structure.metrics import tm_score_coords
from scipy.spatial import cKDTree

from . import regions
from ._warnings import warn
from .angles import angle_settings, docking_angles
from .regions import CDRS, residue_labels

__all__ = ['match', 'superposition', 'superpose', 'SUPERPOSITION_MODES', 'rmsds',
           'docking_rmsds', 'peptide_reversed', 'dockq', 'lddt', 'per_residue_lddt',
           'tm_score', 'angles', 'lddt_coords', 'dockq_contacts', 'BACKBONE', 'MIN_IDENTITY',
           'METRIC_METHODS', 'FIT_ATOMS', 'TCR_CHAINS', 'PMHC_CHAINS']

#: Backbone atoms for iRMSD / LRMSD.
BACKBONE = ('N', 'CA', 'C', 'O')

#: Chains below this sequence identity to the reference are not scored.
MIN_IDENTITY = 0.3

#: Versions of DockQ and TM-score: ``'v1'`` and ``'v2'`` (the default).
METRIC_METHODS = ('v1', 'v2')

#: What the superposition is fitted on: ``'ca'`` (default) or ``'heavy'`` atoms.
FIT_ATOMS = ('ca', 'heavy')

#: The two sides of the DockQ v2 interface.
TCR_CHAINS, PMHC_CHAINS = ('D', 'E'), ('A', 'B', 'C', 'F')

_XYZ = ['x', 'y', 'z']


# ------------------------------------------------------------------- matching

def _identity(seq1: str, seq2: str) -> float:
    """Identical residues per alignment column (gaps count), global alignment."""
    aln = SeqFrame(seq1).align(seq2).df
    return float((aln['match'] == 'identical').sum() / len(aln)) if len(aln) else 0.0


def match(reference: PDBFrame, model: PDBFrame, min_identity: float = MIN_IDENTITY
          ) -> tuple[PDBFrame, PDBFrame, dict[str, float]]:
    """The atoms the two structures share, row for row.

    Chains are paired by ID (both annotated with standard IDs). Each pair's
    sequences are aligned globally; only aligned residues with the *same* amino
    acid are kept, and within them the atoms present in both. A chain whose
    identity (identical residues per alignment column) is below `min_identity`
    is dropped, with a warning.

    Returns:
        ``(reference, model, identity)``: two PDBFrames whose row i is the same
        atom, and ``{chain: identity}`` for every common chain.
    """
    ref_seq, mod_seq = reference.sequences, model.sequences
    common = [c for c in reference.chains if c in mod_seq]
    identity = {c: _identity(ref_seq[c], mod_seq[c]) for c in common}
    kept = {}
    for c in common:
        if identity[c] < min_identity:
            warn(f'{model.name}: chain {c} is {identity[c]:.0%} identical to '
                 f'{reference.name} chain {c} (below {min_identity:.0%}); chain not scored')
        else:
            kept[c] = c
    if not kept:
        raise ValueError(f'no chains of {model.name} match {reference.name}')
    ref, mod = reference.match(model, by='sequence', chains=kept, min_identity=0)
    same = (ref.df['aa'].to_numpy() == mod.df['aa'].to_numpy())
    return ref.select(mask=same), mod.select(mask=same), identity


#: Superposition modes: the region the fit is computed on (its Cα atoms, or with
#: ``atoms='heavy'`` all its matched heavy atoms).
SUPERPOSITION_MODES = {
    'peptide_centric': 'residues of any chain within 10 Å of the reference peptide core (default)',
    'full': 'all matched atoms',
    'pmhc': 'the pMHC: chains A, B, C, F (so TCR errors show as TCR placement)',
    'mhc': 'the MHC chains A and B only (so the peptide placement in the groove counts too)',
    'tcr': 'the TCR chains D and E (so pMHC errors show relative to the TCR)',
    'v_domains': 'the TCR V domains, IMGT 1-128 of D and E (the TCR without its constant domains)',
    'cdrs': 'the six CDRs of the reference (so the CDR RMSDs are of loop shape, not placement)',
}


def superposition(reference: PDBFrame, model: PDBFrame, mode: str = 'peptide_centric',
                  atoms: str = 'ca') -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation ``(R, t)`` that fit the matched model onto the reference.

    Apply them with ``frame.transform(R, t)`` - also to the full, unmatched model.

    Args:
        mode: what to fit on
            (see :data:`SUPERPOSITION_MODES`):

            - ``'peptide_centric'``: every residue within 10 Å of the reference's
              peptide core (all chains, peptide included);
            - ``'full'``: all matched atoms;
            - ``'pmhc'``: chains A, B, C and F;
            - ``'mhc'``: the MHC chains A and B only (no peptide, no B2M);
            - ``'tcr'``: the TCR chains D and E;
            - ``'v_domains'``: the two TCR V domains (IMGT 1-128 of D and E, from
              the reference's labels), leaving out the constant domains;
            - ``'cdrs'``: the six CDRs (IMGT ranges, from the reference's labels).

            Only ``'peptide_centric'`` has saved reference numbers to check against;
            ``'full'`` and ``'pmhc'`` are in the reference code too.
        atoms: ``'ca'`` (default): fit the Cα atoms of that region. ``'heavy'``
            (``'v1'``): fit all its matched heavy atoms.
    """
    if atoms not in FIT_ATOMS:
        raise ValueError(f'atoms must be one of {FIT_ATOMS}, got {atoms!r}')
    if mode == 'peptide_centric':
        if 'C' not in reference.chains:
            raise ValueError(f'{model.name} and {reference.name} share no peptide (chain C), '
                             'which the peptide-centric superposition needs; superpose on '
                             "'full' instead (compare(..., align_on='full'))")
        fit = regions.near_residues(reference, regions.peptide_core(reference), 10.0)
    elif mode == 'full':
        fit = np.ones(len(reference), bool)
    elif mode in ('pmhc', 'mhc', 'tcr'):
        chains = {'pmhc': ['A', 'B', 'C', 'F'], 'mhc': ['A', 'B'], 'tcr': ['D', 'E']}[mode]
        fit = reference.df['chain'].isin(chains).to_numpy()
        if not fit.any():
            raise ValueError(f'{model.name} and {reference.name} share none of the chains '
                             f"{', '.join(chains)}; superpose on 'full' instead "
                             "(compare(..., align_on='full'))")
    elif mode in ('cdrs', 'v_domains'):
        parts = ([regions.cdr(reference, c) for c in CDRS] if mode == 'cdrs' else
                 [regions.v_domain(reference, c) for c in ('D', 'E')])
        fit = np.logical_or.reduce(parts)
        if not fit.any():
            what = 'CDR' if mode == 'cdrs' else 'TCR V-domain'
            raise ValueError(f'{model.name} and {reference.name} share no {what} residues; '
                             "superpose on 'full' instead (compare(..., align_on='full'))")
    else:
        raise ValueError(f'mode (compare: align_on) must be one of '
                         f'{list(SUPERPOSITION_MODES)}, not {mode!r}')
    if atoms == 'ca':
        fit = fit & (reference.df['atom_name'] == 'CA').to_numpy()
    return kabsch(model.coords[fit], reference.coords[fit])


def superpose(reference: PDBFrame, model: PDBFrame, mode: str = 'peptide_centric',
              atoms: str = 'ca') -> PDBFrame:
    """The matched model moved onto the reference; see :func:`superposition`."""
    return model.transform(*superposition(reference, model, mode, atoms))


# ---------------------------------------------------------------------- RMSD

def _rank_paired(ref_df: pd.DataFrame, mod_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pair two selections as ``'v1'`` does: by chain, n-th residue, aa and atom name.

    Used for CDRs, which each structure picks by its own IMGT numbering, so the
    selections need not be the same residues. They must have as many residues.
    """
    def keys(df):
        rank = df.groupby('chain')['observed_index'].rank(method='dense').astype(int)
        return pd.MultiIndex.from_arrays([df['chain'], rank, df['aa'], df['atom_name']])

    n1 = len(ref_df.drop_duplicates(['chain', 'observed_index']))
    n2 = len(mod_df.drop_duplicates(['chain', 'observed_index']))
    if n1 != n2:
        raise ValueError(f'residue counts differ: {n1} in the reference, {n2} in the model')
    k1, k2 = keys(ref_df), keys(mod_df)
    a, b = ref_df[k1.isin(k2)], mod_df[k2.isin(k1)]
    if len(a) != len(b):
        raise ValueError('selections do not pair up atom for atom')
    return a[_XYZ].to_numpy(float), b[_XYZ].to_numpy(float)


def rmsds(reference: PDBFrame, model: PDBFrame) -> dict[str, float]:
    """RMSD (Å) of heavy atoms in the current pose, no refit.

    ``Full`` (all matched atoms), ``Peptide`` (peptide core),
    ``Peptide_Centric`` (residues within 10 Å of the core), one per CDR and
    ``Mean_CDR``. CDRs are picked in each structure by its own IMGT labels and
    paired by rank; a CDR whose residue counts differ
    is NaN, with a warning, and so is ``Mean_CDR``.
    """
    ref_xyz, mod_xyz = reference.coords, model.coords
    core = regions.peptide_core(reference)
    centric = regions.near_residues(reference, core, 10.0)
    out = {'Full': rmsd(mod_xyz, ref_xyz),
           'Peptide': rmsd(mod_xyz[core], ref_xyz[core]),
           'Peptide_Centric': rmsd(mod_xyz[centric], ref_xyz[centric])}
    if {'D', 'E'} <= set(reference.chains):
        for name in CDRS:
            try:
                a, b = _rank_paired(reference.df[regions.cdr(reference, name)],
                                    model.df[regions.cdr(model, name)])
                out[name] = rmsd(b, a)
            except ValueError as e:
                warn(f'{model.name}: {name} RMSD not computed: {e}')
                out[name] = np.nan
        out['Mean_CDR'] = float(np.mean([out[n] for n in CDRS]))
    return out


def docking_rmsds(reference: PDBFrame, model: PDBFrame) -> dict[str, float]:
    """Cα RMSD (Å) of each side of the TCR-pMHC interface, in the current pose, no refit.

    ``TCR_LRMSD``: the TCR (chains D, E). ``pMHC_LRMSD``: the pMHC (A, B, C, F). After
    ``superpose(..., mode='pmhc')`` the first is the pMHC-aligned ligand RMSD (where
    the TCR landed on the pMHC); after
    ``mode='tcr'`` the second is the TCR-aligned one. Cα atoms, as docking L-RMSDs are
    of the backbone. NaN for a side the two structures do not share.
    """
    df = reference.df
    is_ca = (df['atom_name'] == 'CA').to_numpy()
    out = {}
    for name, chains in [('TCR_LRMSD', TCR_CHAINS), ('pMHC_LRMSD', PMHC_CHAINS)]:
        side = is_ca & df['chain'].isin(chains).to_numpy()
        out[name] = rmsd(model.coords[side], reference.coords[side]) if side.any() else np.nan
    return out


def peptide_reversed(reference: PDBFrame, model: PDBFrame) -> bool:
    """True if the model's peptide core runs the other way in the groove from the
    reference's.

    Each structure's direction is its own: the core's first-to-last Cα against
    the N- to C-terminal direction of its MHC alpha1 helix (see
    :func:`.angles.peptide_direction`), so the superposition does not matter.
    """
    from .angles import peptide_direction
    return bool(np.sign(peptide_direction(reference)) != np.sign(peptide_direction(model)))


# ------------------------------------------------------------------- TM-score

def _check_metric_method(method: str, metric: str) -> str:
    if method not in METRIC_METHODS:
        raise ValueError(f'{metric} method must be one of {METRIC_METHODS}, got {method!r}')
    return method


def _first_ca(df: pd.DataFrame) -> np.ndarray:
    """One Cα row per residue."""
    ca = (df['atom_name'] == 'CA').to_numpy()
    first = np.zeros(len(df), bool)
    first[np.flatnonzero(ca)[~df[ca].duplicated(['chain', 'observed_index']).to_numpy()]] = True
    return first


def tm_score(reference: PDBFrame, model: PDBFrame, method: str = 'v2',
             length: int | None = None) -> float:
    """TM-score of the matched Cα atoms, normalised by `length`.

    Args:
        method: ``'v2'`` (default): on their own best fit (Kabsch on all matched Cα
            atoms), so the pose does not matter. ``'v1'``, as the reference
            benchmark: in the current pose.
        length: the L of the score (and of its d0); default: the matched residues.
            Pass the reference's length to count missing residues against the model.
    """
    _check_metric_method(method, 'tm_score')
    ca = _first_ca(reference.df)
    a, b = model.coords[ca], reference.coords[ca]
    if method == 'v2':
        a = transform(a, *kabsch(a, b))
    return tm_score_coords(a, b, length=length or int(ca.sum()), superpose=False)


# ----------------------------------------------------------------------- lDDT

def lddt(reference: PDBFrame, model: PDBFrame, radius: float = 15.0,
         thresholds=(0.5, 1.0, 2.0, 4.0), per_residue: bool = False):
    """lDDT of the matched heavy atoms; see :func:`lddt_coords`.

    With ``per_residue=True``, the reference's residue table with an ``lddt``
    column (NaN for residues without neighbours) - ready for
    ``pf.view(values=...)``.
    """
    df = reference.df
    labels = residue_labels(df)
    score = lddt_coords(model.coords, reference.coords, labels,
                        (df['atom_name'] == 'CA').to_numpy(), radius, thresholds,
                        per_residue=per_residue)
    if not per_residue:
        return score
    res = deposited.observed_residues(reference.df)
    return res.assign(lddt=pd.Series(score).reindex(residue_labels(res)).to_numpy())


def per_residue_lddt(model, reference, by: str | None = None,
                     min_identity: float = MIN_IDENTITY) -> pd.DataFrame | pd.Series:
    """lDDT of each residue of `model` against `reference` (no superposition needed).

    Args:
        model, reference: structure files or frames, as for :func:`.compare.compare`.
        by: ``None`` (default): one row per matched residue, with ``imgt``, ``region``
            (the most specific region the residue is in, ``''`` for none) and
            ``lddt``. ``'region'`` or ``'chain'``: the mean lDDT per region (residues
            in no region left out) or chain, most wrong first.
        min_identity: as for :func:`match`.
    """
    from .chains import scoring_frame
    from .tcr import TCRFrame
    ref_full = TCRFrame.ensure(scoring_frame(reference))
    ref, mod, _ = match(ref_full, scoring_frame(model), min_identity=min_identity)
    res = lddt(ref, mod, per_residue=True)
    keys = ['chain', 'residue_number', 'insertion_code']
    labels = ref_full.residues[keys + ['imgt', 'region']]
    res = res.drop(columns=[c for c in ('imgt', 'region') if c in res]).merge(
        labels, on=keys, how='left')
    cols = [c for c in ['chain', 'residue_number', 'insertion_code', 'aa', 'imgt', 'region',
                        'lddt'] if c in res]
    res = res[cols].reset_index(drop=True)
    if by is None:
        return res
    if by not in ('region', 'chain'):
        raise ValueError(f"by must be None, 'region' or 'chain', got {by!r}")
    if by == 'region':                     # residues in no region have region ''
        res = res[res['region'].fillna('') != '']
    return res.groupby(by)['lddt'].mean().sort_values()


def lddt_coords(model: np.ndarray, ref: np.ndarray, residue: np.ndarray,
                is_ca: np.ndarray, radius: float = 15.0,
                thresholds=(0.5, 1.0, 2.0, 4.0), per_residue: bool = False):
    """lDDT as the reference computes it (``compute_lddt2``).

    Residues are neighbours when their reference Cα atoms are within `radius`.
    Each residue with a Cα scores every heavy-atom pair between itself and its
    neighbours: the share of pairs whose distance the model keeps within each
    threshold (``<=``), averaged over thresholds. The result is the mean over
    residues that have neighbours.

    Args:
        model, ref: paired heavy-atom coordinates.
        residue: residue label per atom.
        is_ca: True for the Cα atom of each residue.
        per_residue: return ``{residue label: score}`` for the residues scored
            instead of the mean.
    """
    codes, uniques = pd.factorize(residue)
    order = np.argsort(codes, kind='stable')          # atoms grouped by residue
    model, ref, codes = model[order], ref[order], codes[order]
    is_ca = np.asarray(is_ca, bool)[order]
    n_res = codes.max() + 1
    count = np.bincount(codes, minlength=n_res)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])

    ca_idx = np.flatnonzero(is_ca)
    _, first = np.unique(codes[ca_idx], return_index=True)   # one Cα per residue
    ca_idx = ca_idx[first]
    pairs = cKDTree(ref[ca_idx]).query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return {} if per_residue else 0.0
    ra, rb = codes[ca_idx[pairs[:, 0]]], codes[ca_idx[pairs[:, 1]]]

    # every atom pair of every neighbouring residue pair
    size = count[ra] * count[rb]
    pair_of = np.repeat(np.arange(len(ra)), size)
    q = np.arange(size.sum()) - np.repeat(np.cumsum(size) - size, size)
    i = start[ra][pair_of] + q // count[rb][pair_of]
    j = start[rb][pair_of] + q % count[rb][pair_of]
    diff = np.abs(np.linalg.norm(model[i] - model[j], axis=1)
                  - np.linalg.norm(ref[i] - ref[j], axis=1))
    kept = (diff[:, None] <= np.asarray(thresholds)).mean(axis=1)

    # each pair counts for both of its residues
    res_a, res_b = ra[pair_of], rb[pair_of]
    total = np.bincount(res_a, kept, n_res) + np.bincount(res_b, kept, n_res)
    n = np.bincount(res_a, minlength=n_res) + np.bincount(res_b, minlength=n_res)
    scored = n > 0
    if per_residue:
        return dict(zip(np.asarray(uniques)[scored], total[scored] / n[scored]))
    return float((total[scored] / n[scored]).mean())


# ---------------------------------------------------------------------- DockQ

def _residue_min_distance(xyz: np.ndarray, residue: np.ndarray, lig: np.ndarray,
                          rec: np.ndarray, cutoff: float) -> dict[tuple, float]:
    """Closest atom distance of every ligand-receptor residue pair closer than `cutoff`."""
    li, ri = np.flatnonzero(lig), np.flatnonzero(rec)
    i, j, d = atom_pairs_within(xyz[li], xyz[ri], cutoff)
    pairs = pd.DataFrame({'a': residue[li][i], 'b': residue[ri][j], 'd': d})
    return pairs.groupby(['a', 'b'])['d'].min().to_dict()


def dockq_contacts(model: np.ndarray, ref: np.ndarray, residue: np.ndarray,
                   ligand: np.ndarray, contact: float = 5.0, clash: float = 2.0) -> dict:
    """Contact counts between ligand and receptor residues (closest atoms < `contact` Å).

    Returns ``fnat``, ``nat_correct``, ``nat_total``, ``fnonnat``, ``nonnat_count``,
    ``model_total``, ``F1``, ``clashes`` (model pairs closer than `clash`), and
    ``len1`` / ``len2``: ligand and receptor residue counts.
    """
    ligand = np.asarray(ligand, bool)
    native = {p for p, d in _residue_min_distance(ref, residue, ligand, ~ligand, contact).items()
              if d < contact}
    mod_d = _residue_min_distance(model, residue, ligand, ~ligand, contact)
    predicted = {p for p, d in mod_d.items() if d < contact}
    correct, nonnat = len(native & predicted), len(predicted - native)
    denom = correct + nonnat + len(native)
    return {
        'fnat': correct / len(native) if native else 0,
        'nat_correct': correct,
        'nat_total': len(native),
        'fnonnat': nonnat / len(predicted) if predicted else 0,
        'nonnat_count': nonnat,
        'model_total': len(predicted),
        'F1': 2 * correct / denom if denom else 0,
        'clashes': sum(d < clash for d in mod_d.values()),
        'len1': len(np.unique(residue[ligand])),
        'len2': len(np.unique(residue[~ligand])),
    }


def dockq(reference: PDBFrame, model: PDBFrame, method: str = 'v2') -> dict:
    """DockQ and its parts: ``DockQ``, ``iRMSD``, ``LRMSD`` and the contact counts of
    :func:`dockq_contacts`.

    Args:
        method: ``'v2'`` (default), as DockQ v2 (see the module docstring): takes the
            two structures as they are (matched or not; it pairs residues and
            superposes by itself). ``'v1'``: the peptide
            core against the MHC around it plus the CDRs, on structures from
            :func:`match` in their current pose.
    """
    if _check_metric_method(method, 'dockq') == 'v1':
        return _dockq_v1(reference, model)
    return _dockq_v2(reference, model)


def _dockq_v1(reference: PDBFrame, model: PDBFrame) -> dict:
    """DockQ of the peptide core against the MHC around it plus the CDRs, no refit.

    The interface is cut down, by the reference, to the peptide core (ligand)
    and the receptor: MHC residues (chains A, B) within 10 Å of the core, and
    all CDR residues. iRMSD is over backbone atoms of residues within 8 Å of the
    ligand, LRMSD over the ligand backbone, both in the current pose; contacts
    are residue pairs closer than 5 Å.
    """
    df = reference.df
    core = regions.peptide_core(reference)
    mhc = df['chain'].isin(['A', 'B']).to_numpy()
    cdrs = np.logical_or.reduce([regions.cdr(reference, n) for n in CDRS])
    trim = core | regions.near_residues(reference, core, 10.0, within=mhc) | cdrs

    ref_t, mod_t = reference.select(mask=trim), model.select(mask=trim)
    ligand = (ref_t.df['chain'] == 'C').to_numpy()
    backbone = ref_t.df['atom_name'].isin(BACKBONE).to_numpy()
    interface = regions.near_residues(ref_t, ligand, 8.0) & backbone
    ref_xyz, mod_xyz = ref_t.coords, mod_t.coords
    irmsd = rmsd(mod_xyz[interface], ref_xyz[interface])
    lig_bb = ligand & backbone
    lrmsd = rmsd(mod_xyz[lig_bb], ref_xyz[lig_bb])

    contacts = dockq_contacts(mod_xyz, ref_xyz, residue_labels(ref_t.df), ligand)
    return {'DockQ': _dockq_score(contacts['fnat'], irmsd, lrmsd), 'iRMSD': irmsd,
            'LRMSD': lrmsd, **contacts}


def _dockq_score(fnat: float, irmsd: float, lrmsd: float) -> float:
    return (fnat + 1 / (1 + (irmsd / 1.5) ** 2) + 1 / (1 + (lrmsd / 8.5) ** 2)) / 3


def _dockq_aligner():
    """DockQ v2's sequence alignment (``DockQ.align_chains``)."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.match, aligner.mismatch = 5, 0
    aligner.open_gap_score, aligner.extend_gap_score = -4, -0.5
    return aligner


def _side_residues(pf: PDBFrame, chains) -> pd.DataFrame:
    """The residues of `chains`, in that chain order and file order, as one sequence."""
    df = pf.df[pf.df['chain'].isin(chains)]
    res = df.drop_duplicates(['chain', 'observed_index'])[['chain', 'observed_index', 'aa']]
    order = {c: i for i, c in enumerate(chains)}
    return res.iloc[np.argsort(res['chain'].map(order).to_numpy(), kind='stable')]


def _paired_residues(ref_res: pd.DataFrame, mod_res: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the residues DockQ pairs: identical amino acids aligned to each other."""
    a, b = ''.join(ref_res['aa']), ''.join(mod_res['aa'])
    if a == b:
        return np.arange(len(a)), np.arange(len(b))
    aln = _dockq_aligner().align(b, a)[0]
    i_ref, i_mod = [], []
    for (m0, m1), (r0, r1) in zip(*aln.aligned):
        for k in range(m1 - m0):
            if b[m0 + k] == a[r0 + k]:
                i_mod.append(m0 + k)
                i_ref.append(r0 + k)
    return np.array(i_ref, int), np.array(i_mod, int)


def _labelled_atoms(pf: PDBFrame, residues: pd.DataFrame, labels: np.ndarray):
    """Atoms of `residues` (one row per residue, labelled by `labels`): coordinates,
    the label of each atom, and the atom names."""
    lab = pd.Series(labels, index=pd.MultiIndex.from_frame(residues[['chain', 'observed_index']]))
    atom = lab.reindex(pd.MultiIndex.from_frame(pf.df[['chain', 'observed_index']]))
    kept = atom.notna().to_numpy()
    return (pf.coords[kept], atom[kept].to_numpy(int),
            pf.df.loc[kept, 'atom_name'].to_numpy())


def _common_backbone(ref, mod, residues) -> tuple[np.ndarray, np.ndarray]:
    """Backbone atoms both structures have in `residues`, paired row for row."""
    def table(xyz, lab, name):
        bb = np.isin(name, BACKBONE) & np.isin(lab, list(residues))
        return pd.DataFrame(xyz[bb], columns=_XYZ,
                            index=pd.MultiIndex.from_arrays([lab[bb], name[bb]])) \
            .groupby(level=[0, 1], sort=True).first()
    a, b = table(*ref), table(*mod)
    both = a.index.intersection(b.index)
    return a.loc[both].to_numpy(float), b.loc[both].to_numpy(float)


def _dockq_v2(reference: PDBFrame, model: PDBFrame) -> dict:
    """DockQ v2 of the TCR against the pMHC; see :func:`dockq`.

    As the DockQ program run on the two sides, each made one chain: residues are
    paired by DockQ's own alignment of each side's sequence; contacts (< 5 Å),
    clashes (< 2 Å) and the interface (< 10 Å) use each structure's own atoms of
    the paired residues; iRMSD and LRMSD use the backbone atoms both have. The
    side with more reference residues is the receptor. ``len1`` / ``len2``: the
    reference residues of the TCR and the pMHC.
    """
    common = set(reference.chains) & set(model.chains)
    sides = [[c for c in TCR_CHAINS if c in common], [c for c in PMHC_CHAINS if c in common]]
    if not all(sides):
        raise ValueError('DockQ v2 needs TCR chains (D, E) and pMHC chains (A, B, C, F) '
                         'in both structures')
    ref_atoms, mod_atoms, ligand_of, n_side = [], [], {}, []
    offset = 0
    for side, chains in enumerate(sides):
        ref_res, mod_res = _side_residues(reference, chains), _side_residues(model, chains)
        n_side.append(len(ref_res))
        i_ref, i_mod = _paired_residues(ref_res, mod_res)
        labels = offset + np.arange(len(i_ref))
        ref_atoms.append(_labelled_atoms(reference, ref_res.iloc[i_ref], labels))
        mod_atoms.append(_labelled_atoms(model, mod_res.iloc[i_mod], labels))
        ligand_of.update(dict.fromkeys(labels, side))
        offset += len(i_ref)
    ref = tuple(np.concatenate(x) for x in zip(*ref_atoms))
    mod = tuple(np.concatenate(x) for x in zip(*mod_atoms))

    def tcr_side(lab):
        return np.array([ligand_of[k] == 0 for k in lab])
    ref_tcr, mod_tcr = tcr_side(ref[1]), tcr_side(mod[1])
    native = _residue_min_distance(ref[0], ref[1], ref_tcr, ~ref_tcr, 10.0)
    model_d = _residue_min_distance(mod[0], mod[1], mod_tcr, ~mod_tcr, 5.0)
    nat = {p for p, d in native.items() if d < 5.0}
    pred = {p for p, d in model_d.items() if d < 5.0}
    correct, nonnat = len(nat & pred), len(pred - nat)

    interface = {r for pair in native for r in pair}
    a, b = _common_backbone(ref, mod, interface)
    irmsd = rmsd(b, a, superpose=True)

    receptor = 0 if n_side[0] > n_side[1] else 1
    rec = [k for k, s in ligand_of.items() if s == receptor]
    lig = [k for k, s in ligand_of.items() if s != receptor]
    rec_ref, rec_mod = _common_backbone(ref, mod, rec)
    lig_ref, lig_mod = _common_backbone(ref, mod, lig)
    lrmsd = rmsd(transform(lig_mod, *kabsch(rec_mod, rec_ref)), lig_ref)

    fnat = correct / len(nat) if nat else 0.0
    return {
        'DockQ': _dockq_score(fnat, irmsd, lrmsd), 'iRMSD': irmsd, 'LRMSD': lrmsd,
        'fnat': fnat, 'nat_correct': correct, 'nat_total': len(nat),
        'fnonnat': nonnat / len(pred) if pred else 0.0, 'nonnat_count': nonnat,
        'model_total': len(pred),
        'F1': 2 * correct / (correct + nonnat + len(nat)) if nat or pred else 0.0,
        'clashes': sum(d < 2.0 for d in model_d.values()),
        'len1': n_side[0], 'len2': n_side[1],
    }


# --------------------------------------------------------------------- angles

def _angles_or_nan(pf: PDBFrame, method: str | dict) -> dict[str, float]:
    try:
        return docking_angles(pf, method=method)
    except Exception as e:  # the reference reports NaN for any failure
        warn(f'{pf.name}: docking angles not computed: {e}', stacklevel=4)
        return {'crossing_angle': np.nan, 'incident_angle': np.nan}


def angles(reference: PDBFrame, model: PDBFrame, method: str | dict = 'piercelab') -> dict[str, float]:
    """The model's crossing / incident angles and their absolute errors (NaN on failure).

    Args:
        method: ``'piercelab'`` (default) or ``'v1'`` (Motif Atlas
            helices, whole-chain axis, 0-90), or a dict of parts such as
            ``{'helices': 'v1'}``; see :func:`.angles.angle_settings`.
    """
    method = angle_settings(method)
    ref, mod = _angles_or_nan(reference, method), _angles_or_nan(model, method)
    return {**mod,
            'crossing_angle_error': abs(ref['crossing_angle'] - mod['crossing_angle']),
            'incident_angle_error': abs(ref['incident_angle'] - mod['incident_angle'])}
