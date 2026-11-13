"""3D views of TCR-pMHC structures, on top of PDBFrame's viewer.

    >>> tcs.view(pdb1)                              # peptide core, CDRs, MHC helices marked
    >>> tcs.view(pdb1, regions='cdr3')              # only CDR3 alpha / beta
    >>> tcs.view(pdb1, regions='all', hidden_regions=['peptide_centric', 'Va', 'Vb'])
    >>> tcs.view(pdb1, vectors=True)               # the docking-angle vectors, with the angles
    >>> tcs.view(pdb1, vectors='v1')               # ... the 'v1' vectors
    >>> tcs.view_comparison(model, reference)      # model superposed, colored by lDDT
    >>> tcs.view_comparison({'AF3': m1, 'Boltz': m2}, reference)

``pdb1.view(...)`` (a TCRFrame) is the same as ``tcs.view(pdb1, ...)``. Region names and
groups are in :data:`.tcr.REGIONS` / :data:`.tcr.REGION_GROUPS`; anything else
goes to :meth:`PDBFrame.view` (``sequence=``, ``style=``, ``surface=``,
``hidden_regions=``, ...).

In the viewer, click a region in the legend to switch it off and on (*CDRs*
switches all of them). The atoms are shown where they are in the file. The *TCR up*
/ *Groove from above* buttons turn the camera to the TCR-up orientation (TCR up,
peptide left to right, MHC below; see :func:`.angles.tcr_up_orientation`) or
onto the groove; ``tcr_up=True`` opens the view turned TCR up. Only the camera
turns: the coordinates are never changed.

``vectors=`` draws the vectors behind the docking angles (see
:func:`.angles.docking_vectors`): the groove's long axis and plane normal from the
helix centre, and the TCR disulfide vector and rotation axis through the middle
of the TCR, labelled with the crossing and incident angles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure.geometry import transform as apply_transform

from ._warnings import warn
from .angles import docking_angles, docking_vectors, orientations
from .chains import annotate, scoring_frame
from .regions import CDRS
from .tcr import _LOOKUP, REGION_COLORS, _key, expand_regions

__all__ = ['view', 'view_comparison', 'vector_arrows', 'VECTOR_COLORS']

_GROUPS = {'CDRs': CDRS, 'CDR alpha': CDRS[:3], 'CDR beta': CDRS[3:]}

#: Colors of the docking-angle vectors.
VECTOR_COLORS = {'groove_axis': '#1F77B4', 'groove_normal': '#2CA02C',
                 'tcr_vector': '#D62728', 'rotation_axis': '#9467BD'}
_VECTOR_LENGTH = 20.0   # Å, half the drawn length of the axes


def _marks(pf: PDBFrame, regions, extra_regions, warn_missing: bool = True) -> dict:
    """``{name: PDBFrame}`` of the named regions of `pf`, plus `extra_regions`."""
    names = [] if regions is None or regions is False else expand_regions(regions)
    found = pf.regions
    missing = [n for n in names if n not in found]
    if missing and warn_missing:
        warn(f'{pf.name}: no {missing} to mark', stacklevel=4)
    marks = {n: _plain(found[n]) for n in names if n in found}
    marks.update(extra_regions or {})
    return marks


def _complex(pf: PDBFrame) -> PDBFrame:
    """The whole complex a selection was taken from (its atoms are where the
    selection's are), or `pf` itself."""
    frame = pf
    while True:
        parent = frame.__dict__.get('_parent')
        if parent is None or parent[1] is None:     # not a selection: own coordinates
            return frame
        frame = parent[0]


def _hidden_names(kwargs: dict) -> None:
    """``hidden_regions=`` by region name in any case (``'cdra1'``, ``'tcra'``), as the
    marks are named; other names (``extra_regions``) are kept as they are."""
    if kwargs.get('hidden_regions'):
        kwargs['hidden_regions'] = [_LOOKUP.get(_key(n), n) for n in kwargs['hidden_regions']]


def _plain(pf: PDBFrame) -> PDBFrame:
    """`pf` as a plain PDBFrame, for PDBFrame's viewer: it reads ``residues``, which on a
    TCRFrame would find the regions of every fragment it is given."""
    return PDBFrame._wrap(pf.df, pf.name) if isinstance(pf, PDBFrame) else pf


def _legend(names, region_colors) -> tuple[dict, dict]:
    """Default colors for the marked regions, and CDR groups for the legend."""
    colors = {n: REGION_COLORS[n] for n in names if n in REGION_COLORS}
    colors.update(region_colors or {})
    groups = {g: [n for n in members if n in names] for g, members in _GROUPS.items()}
    return colors, {g: m for g, m in groups.items() if len(m) > 1}


def _orientations(pf: PDBFrame, tcr_up: bool, kwargs: dict) -> None:
    """Add the *TCR up* / *Groove from above* buttons to `kwargs`, and with `tcr_up`
    open the view turned TCR up (PDBFrame's ``orientation=``: only the camera turns).
    A selection is oriented by the complex it came from."""
    try:
        kwargs.setdefault('orientations', orientations(_complex(pf)))
    except Exception as e:
        what = 'not shown TCR up, and no' if tcr_up else 'no'
        warn(f'{pf.name}: TCR-up orientation not found ({e}); {what} orientation '
             'buttons', stacklevel=4)
        return
    if tcr_up:
        kwargs.setdefault('orientation', 'TCR up')


def vector_arrows(pf: PDBFrame, method: str | dict = 'piercelab', transform=None) -> list[dict]:
    """The docking-angle vectors of `pf` as arrows for ``PDBFrame.view(arrows=...)``.

    Args:
        method: how the angles are computed (see :func:`.angles.angle_settings`).
        transform: ``(R, t)`` applied to the points, to draw on a transformed copy.
    """
    v = docking_vectors(pf, method)
    a = docking_angles(pf, v['settings'])
    signed = v['settings']['signed']
    L, gc, tc = _VECTOR_LENGTH, v['groove_center'], v['tcr_center']
    tcr_label = 'TCR alpha->beta' if signed else 'TCR beta->alpha'
    spans = {'groove_axis': (gc - L * v['groove_axis'], gc + L * v['groove_axis'], 'groove axis'),
             'groove_normal': (gc, gc + L * v['groove_normal'], 'groove normal'),
             'tcr_vector': (tc - L / 2 * v['tcr_vector'], tc + L / 2 * v['tcr_vector'],
                            f"{tcr_label}: crossing {a['crossing_angle']:.1f}°"),
             'rotation_axis': (tc - L * v['rotation_axis'], tc + L * v['rotation_axis'],
                               f"rotation axis: incident {a['incident_angle']:.1f}°")}
    move = (lambda xyz: xyz) if transform is None else \
        (lambda xyz: apply_transform(xyz, *map(np.asarray, transform)))
    return [{'start': move(start), 'end': move(end), 'color': VECTOR_COLORS[k],
             'label': label, 'radius': 0.7, 'group': 'Docking vectors'}
            for k, (start, end, label) in spans.items()]


def view(structure, regions=True, extra_regions: dict | None = None,
         region_colors: dict | None = None, tcr_up: bool = False, vectors=False, **kwargs):
    """A TCR-pMHC in 3D with its regions marked, its atoms where they are in the file.

    Args:
        structure: a file or PDBFrame (annotated first if needed, so the view
            shows the standard chain IDs).
        regions: which regions to mark - ``True`` (the default set: peptide core,
            the six CDRs, MHC helices), ``'all'``, a region or group name
            (``'cdrs'``, ``'cdr3'``, ``'tcra'``, ...), a list of these, or ``None``.
            Each can be switched off and on in the viewer's legend.
        extra_regions: more regions to mark, ``{name: spec}`` as for
            :meth:`PDBFrame.view` (a PDBFrame selection, a chain ID, a
            ``(chain, start, end)`` range).
        region_colors: ``{name: color}`` overriding the default colors.
        tcr_up: open the view turned to the TCR-up orientation (TCR up, peptide
            left to right). Only the camera turns; the coordinates stay as in the
            file. The *TCR up* / *Groove from above* buttons are there either way.
        vectors: draw the docking-angle vectors with the angles - ``True`` (the
            default method, ``'piercelab'``), ``'v1'``, or a dict of parts such
            as ``{'helices': 'v1'}``. The marked ``helices`` region then shows
            the helix residues that method uses. A checkbox switches them off.
        kwargs: passed to :meth:`PDBFrame.view`, e.g. ``hidden_regions=``
            (regions to start switched off), ``sequence=``, ``style=``.
    """
    pf = annotate(structure)
    # a selection has only some of the default regions: mark those, without a warning
    selection = _complex(pf) is not pf
    marks = _marks(pf, regions, extra_regions, warn_missing=not (regions is True and selection))
    if vectors is not False and vectors is not None:
        method = 'piercelab' if vectors is True else vectors
        try:
            kwargs['arrows'] = vector_arrows(pf, method) + list(kwargs.get('arrows') or [])
            if 'helices' in marks:
                marks['helices'] = _plain(pf.select(mask=docking_vectors(pf, method)['helices']))
        except Exception as e:
            warn(f'{pf.name}: docking vectors not drawn: {e}', stacklevel=3)
    colors, groups = _legend(list(marks), region_colors)
    _hidden_names(kwargs)
    _orientations(pf, tcr_up, kwargs)
    kwargs.setdefault('region_groups', groups)
    return PDBFrame.view(_plain(pf), regions=marks, region_colors=colors, **kwargs)


def _model_lddt(reference: PDBFrame, model: PDBFrame) -> pd.DataFrame:
    """The model's residue table with its per-residue lDDT against the reference."""
    from . import metrics
    r, m, _ = metrics.match(scoring_frame(reference), scoring_frame(model))
    per_ref = metrics.lddt(r, m, per_residue=True)
    # matched rows pair residue by residue, so carry the scores over to the model's numbering
    keys = ['residue_number', 'insertion_code']
    pairs = pd.concat([r.df[['chain', *keys]], m.df[keys].add_prefix('model_')], axis=1)
    pairs = pairs.drop_duplicates(['chain', *keys])
    scored = pairs.merge(per_ref[['chain', *keys, 'lddt']], on=['chain', *keys])
    return pd.DataFrame({'chain': scored['chain'],
                         'residue_number': scored['model_residue_number'],
                         'insertion_code': scored['model_insertion_code'],
                         'lddt': scored['lddt']})


def view_comparison(models, reference, superpose=True, values: str | None = 'lddt',
                    regions=True, extra_regions: dict | None = None,
                    region_colors: dict | None = None, tcr_up: bool = False, **kwargs):
    """The reference with one or more models drawn in the same space.

    Uses :meth:`PDBFrame.view` with ``compare=``: residues are linked across all
    structures (hovering one lights up its partners), each model gets its own
    color and checkbox, and their sequences line up under the reference's.

    Args:
        models: a model (file or PDBFrame), a list of them, or ``{name: model}``.
        reference: file or PDBFrame; it leads the view.
        superpose: passed to :meth:`PDBFrame.view` - ``True`` (default) fits each
            model on the reference by Cα atoms of all chains, a chain ID or list
            (e.g. ``['A', 'C']`` for the pMHC) fits on those only, ``False`` draws
            them as they are in their files (predicted models usually sit tens of
            Å away). This is the viewer's fit, not the benchmark's peptide-centric
            one, so the picture can differ slightly from what the RMSDs measure.
        values: ``'lddt'`` colors each model by its per-residue lDDT against the
            reference (red low, blue high; lDDT needs no superposition); ``None``
            keeps the structure colors.
        regions: regions to mark, in the reference and every model (each by its
            own numbering, so a region is switched on and off in all at once).
        tcr_up: open the view turned so the reference is TCR up (only the camera
            turns).
        kwargs: passed to :meth:`PDBFrame.view` (``sequence=``, ``hidden_regions=``, ...).
    """
    ref = annotate(reference)
    if isinstance(models, dict):
        names = [ref.name, *models]
        models = list(models.values())
    else:
        models = list(models) if isinstance(models, (list, tuple)) else [models]
        names = None
    models = [annotate(m) for m in models]

    marks = [_marks(ref, regions, extra_regions)] + \
        [_marks(m, regions, None, warn_missing=False) for m in models]
    vals = None
    if values == 'lddt':
        vals = [None] + [_model_lddt(ref, m) for m in models]
    elif values is not None:
        vals = values
    colors, groups = _legend(list(dict.fromkeys(k for d in marks for k in d)), region_colors)
    _hidden_names(kwargs)
    _orientations(ref, tcr_up, kwargs)
    kwargs.setdefault('region_groups', groups)
    if names is not None:
        kwargs.setdefault('names', names)
    return PDBFrame.view(_plain(ref), regions=marks, region_colors=colors,
                         compare=[_plain(m) for m in models], values=vals,
                         superpose=superpose, **kwargs)
