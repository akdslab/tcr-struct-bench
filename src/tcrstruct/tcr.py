"""``TCRFrame``: a PDBFrame of a TCR-pMHC complex, annotated, with everything TCR-pMHC
about it computed on first use.

    >>> from tcrstruct import TCRFrame
    >>> pdb1 = TCRFrame.read('7n2n.pdb')       # chains A-F, IMGT labels
    >>> pdb1 = TCRFrame.from_rcsb('7N2N')      # the same, straight from the RCSB PDB
    >>> pdb1                                   # name, MHC class, chains
    >>> print(pdb1.report())                   # + genes, peptide and core, CDRs
    >>> pdb1.df                                # the atom table
    >>> pdb1.residues                          # residue table with 'imgt' and 'region'
    >>> pdb1.chain_table                       # what each chain is, gene / allele, sequence
    >>> pdb1.species, pdb1.mhc_species         # the TCR's species (the complex's), the MHC's
    >>> pdb1.cdrs, pdb1.sequence('peptide_core'), pdb1.angles
    >>> pdb1.contact_map()                     # CDR residues touching the peptide
    >>> pdb1.docking_angles(method='v1')            # the 'v1' angles
    >>> pdb1.select(regions='cdr3')            # region atoms, as a TCRFrame
    >>> pdb1.select(chains='D').residues       # selections are TCRFrames too
    >>> pdb1.view()                            # 3D, with the peptide core, CDRs and helices marked
    >>> pdb1.verify()                          # complete complex? chains, peptide, contacts
    >>> pdb1.split_fused_peptide().verify()    # e.g. 6XCP: peptide fused to MHC beta

A TCRFrame is always annotated (:func:`.chains.annotate`): standard chain IDs
(A-F), and ``imgt`` and ``original_chain`` columns. ``TCRFrame(pf)`` annotates a
PDBFrame you already have. Every PDBFrame method works and returns a TCRFrame.
"""

from __future__ import annotations

import html
import re
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pdbframe import PDBFrame
from pdbframe.structure import deposited
from pdbframe.structure.selection import select_mask
from tcrkit.mhc import mhc_species

from . import mhc as _mhc
from . import regions as _regions
from ._warnings import warn
from .angles import angle_settings, docking_angles, docking_vectors
from .chains import annotate, normalize
from .numbering import CDR_IMGT, run_anarci
from .regions import CDRS

__all__ = ['TCRFrame', 'REGIONS', 'REGION_GROUPS', 'REGION_COLORS', 'list_regions',
           'list_region_groups']

#: Every region ``pf.regions`` can hold, in display order. Names are
#: case-insensitive in ``select(regions=...)`` and ``view(regions=...)``.
REGIONS = ['peptide', 'peptide_core', 'peptide_centric', *CDRS, 'helices', 'Va', 'Vb',
           'TCRa', 'TCRb', 'MHC', 'B2M']

#: Names that stand for several regions.
REGION_GROUPS = {
    'default': ['peptide_core', *CDRS, 'helices'],
    'all': ['peptide', 'peptide_core', 'peptide_centric', *CDRS, 'helices', 'Va', 'Vb'],
    'cdrs': CDRS,
    'cdr1': ['CDRA1', 'CDRB1'],
    'cdr2': ['CDRA2', 'CDRB2'],
    'cdr3': ['CDRA3', 'CDRB3'],
    'cdras': ['CDRA1', 'CDRA2', 'CDRA3'],
    'cdrbs': ['CDRB1', 'CDRB2', 'CDRB3'],
    'tcr': ['TCRa', 'TCRb'],
    'pmhc': ['MHC', 'B2M', 'peptide'],
}

#: Region colors: alpha CDRs in blues, beta CDRs in reds, the peptide in yellow.
REGION_COLORS = {
    'peptide': '#ffe066', 'peptide_core': '#f5b800', 'peptide_centric': '#b5d99c',
    'CDRA1': '#9ecae1', 'CDRA2': '#4a90c8', 'CDRA3': '#0b4f8a',
    'CDRB1': '#fcbba1', 'CDRB2': '#f0624d', 'CDRB3': '#a50f15',
    'helices': '#b8b8b8', 'Va': '#dbe9f6', 'Vb': '#fde0d7',
    'TCRa': '#c6dbef', 'TCRb': '#fcd0c0', 'MHC': '#d9d9d9', 'B2M': '#e5d8bd',
}

_ROLES = {'A': 'MHC heavy chain', 'B': 'MHC beta', 'C': 'peptide', 'D': 'TCR alpha',
          'E': 'TCR beta', 'F': 'B2M'}

_DESCRIPTIONS = {
    'peptide': 'the whole peptide (chain C)',
    'peptide_core': ('the whole peptide (class I) or its 9-mer closest to the MHC (class II); '
                     'Peptide, the DockQ ligand'),
    'peptide_centric': ('residues of any chain within 10 A of the peptide core: the '
                        'superposition region, Peptide_Centric RMSD'),
    **{c: f'IMGT CDR{c[-1]} of TCR {"alpha" if c[3] == "A" else "beta"}' for c in CDRS},
    'helices': "the MHC groove helices (fixed IMGT positions, as the default angles)",
    'Va': 'TCR alpha V domain (IMGT 1-128)',
    'Vb': 'TCR beta V domain (IMGT 1-128)',
    'TCRa': 'the whole TCR alpha chain (D)',
    'TCRb': 'the whole TCR beta chain (E)',
    'MHC': 'the MHC chains: A, and B for class II',
    'B2M': 'beta-2 microglobulin (F, class I)',
}

_CHAIN_REGIONS = {'TCRa': ['D'], 'TCRb': ['E'], 'MHC': ['A', 'B'], 'B2M': ['F']}


def _helix_positions() -> str:
    """The groove-helix IMGT positions per G domain, from :func:`tcrkit.imgt_regions`."""
    from tcrkit import imgt_regions
    rows = imgt_regions('mhc')
    return '; '.join(f"{r.domain} {r.imgt_start}-{r.imgt_end}"
                     + (f' without {r.skipped}' if r.skipped else '')
                     for r in rows.itertuples())


def list_regions() -> pd.DataFrame:
    """Every region a TCRFrame can select, and how it is defined.

    Returns:
        One row per region (the index, as in :data:`REGIONS`): ``chains``,
        ``defined_by`` (``'IMGT positions'``, the same in every structure;
        ``'distance'``, from the coordinates; ``'whole chain'``), ``imgt`` (the IMGT
        positions, for regions defined by them) and ``description``. The groups of
        regions are :func:`list_region_groups`.
    """
    cdr = {c: f'{CDR_IMGT[int(c[-1])][0]}-{CDR_IMGT[int(c[-1])][1]}' for c in CDRS}
    rows = {
        'peptide': ('C', 'whole chain', '-'),
        'peptide_core': ('C', 'whole chain (class I); distance to the MHC (class II)', '-'),
        'peptide_centric': ('any', 'distance: within 10 A of the peptide core', '-'),
        **{c: ('D' if c[3] == 'A' else 'E', 'IMGT positions', cdr[c]) for c in CDRS},
        'helices': ('A, B', 'IMGT positions', _helix_positions()),
        'Va': ('D', 'IMGT positions', '1-128'),
        'Vb': ('E', 'IMGT positions', '1-128'),
        'TCRa': ('D', 'whole chain', '-'),
        'TCRb': ('E', 'whole chain', '-'),
        'MHC': ('A, B', 'whole chain', '-'),
        'B2M': ('F', 'whole chain', '-'),
    }
    out = pd.DataFrame.from_dict(rows, orient='index', columns=['chains', 'defined_by', 'imgt'])
    out['description'] = pd.Series(_DESCRIPTIONS)
    out.index.name = 'region'
    return out.loc[REGIONS]


def list_region_groups() -> pd.DataFrame:
    """The names that stand for several regions (:data:`REGION_GROUPS`): one row per
    group, with its ``regions``. Any of them works in ``select(regions=...)`` and
    ``view(regions=...)``."""
    return pd.DataFrame({'regions': {g: ', '.join(m) for g, m in REGION_GROUPS.items()}}
                        ).rename_axis('group')


@lru_cache(maxsize=1024)
def _germline(sequence: str) -> tuple[str, float, str, float, str]:
    """ANARCI's closest V and J genes of a TCR chain, its identity to each, and their
    species (``'-'`` and NaN if not numbered)."""
    n = run_anarci({'chain': sequence})['chain']
    return ((n.v_gene, n.v_identity, n.j_gene, n.j_identity, n.species) if n
            else ('-', np.nan, '-', np.nan, '-'))


def _key(name: str) -> str:
    """Case and separators do not matter: 'CDRA1', 'cdra1' and 'cdr_a1' are one name."""
    return re.sub(r'[\s_\-]', '', str(name)).lower()


_LOOKUP = {_key(n): n for n in REGIONS}
_GROUP_LOOKUP = {_key(g): members for g, members in REGION_GROUPS.items()}


def expand_regions(regions) -> list[str]:
    """Region names from ``True``, a region / group name, or a list of these.

    Matching ignores case, spaces, ``_`` and ``-``: ``'cdra1'`` is ``'CDRA1'``,
    ``'tcra'`` is ``'TCRa'``.
    """
    if regions is True:
        regions = 'default'
    names = [regions] if isinstance(regions, str) else list(regions)
    out = []
    for n in names:
        k = _key(n)
        if k in _GROUP_LOOKUP:
            out.extend(_GROUP_LOOKUP[k])
        elif k in _LOOKUP:
            out.append(_LOOKUP[k])
        else:
            raise KeyError(f'unknown region {n!r}; regions are {REGIONS}, groups are '
                           f'{list(REGION_GROUPS)} (case does not matter)')
    return list(dict.fromkeys(out))


class TCRFrame(PDBFrame):
    """A TCR-pMHC structure: an annotated PDBFrame with its chains, regions, CDRs and
    docking angles.

    Args:
        data: a PDBFrame or an atom table; annotated if it is not yet (chain
            detection, ANARCI). Use :meth:`read` / :meth:`from_rcsb` for files.
        name: a label; default the input's.

    Everything TCR-pMHC is computed when first asked for and then kept, so treat
    a TCRFrame as immutable (every method returns a new one). Parts that cannot be
    computed (no TCR, an unknown MHC, ...) are left out with a
    :class:`TCRStructWarning`.
    """

    def __init__(self, data, name: str | None = None, complex: int | None = 1):
        frame = annotate(data if isinstance(data, PDBFrame) else PDBFrame(data, name=name),
                         name=name, complex=complex)
        self.df, self.name, self.metadata = frame.df, frame.name, frame.metadata
        self._seqres = frame._seqres

    # -------------------------------------------------------------- building

    @classmethod
    def read(cls, path: str | Path, model: int = 0, hetero: bool = False,
             name: str | None = None, complex: int | None = 1) -> TCRFrame:
        """Read a TCR-pMHC structure file and annotate it; see :meth:`PDBFrame.read`.

        Args:
            complex: for a file with several copies of the complex, which one to keep,
                found by contacts (from 1; :func:`.complexes.complex_table` shows how
                the chains group), with a warning naming the chains left out.
                ``None`` keeps every chain, the first of each kind by chain ID
                getting its standard ID (which can mix copies).
        """
        return annotate(PDBFrame.read(path, model=model, hetero=hetero, name=name),
                        complex=complex)

    @classmethod
    def from_rcsb(cls, pdb_id: str, assembly: int | None = 1, save: str | Path | None = None,
                  complex: int | None = 1, **kwargs) -> TCRFrame:
        """An RCSB entry, downloaded and annotated: by default one complex of its first
        biological assembly.

        Two steps, each with a switch:

        1. ``assembly=1`` (default): RCSB's biological assembly 1, what the depositors
           call the biological unit; ``None``: the deposited structure (the
           asymmetric unit), which often holds several copies. An entry without an
           assembly file falls back to the deposited structure, with a warning.
        2. ``complex=1`` (default): of what was downloaded, one complex, found by
           contacts (an assembly can hold more than one: 9RUP's is a dimer);
           ``complex=2`` for the next; ``None`` keeps every chain.

        ``assembly=None, complex=None`` gives the whole entry as deposited.

        Args:
            pdb_id: the 4-character PDB ID.
            assembly: the biological assembly number, or ``None``.
            save: also keep the downloaded file (a folder or a file path).
            complex: which copy of the complex to keep, as for :meth:`read`.
            kwargs: passed to :meth:`PDBFrame.from_rcsb` (``fmt=``, ``hetero=``,
                ``metadata=True`` for the entry's title, release date, resolution, ...
                in ``s.metadata``). ``s.metadata['assembly']`` records the assembly.
        """
        try:
            pf = PDBFrame.from_rcsb(pdb_id, assembly=assembly, save=save, **kwargs)
        except ValueError as e:
            if assembly is None or 'HTTP 404' not in str(e):
                raise
            warn(f'{pdb_id.upper()}: RCSB has no biological assembly {assembly}, so the '
                 'deposited structure is used (assembly=None)')
            assembly = None
            pf = PDBFrame.from_rcsb(pdb_id, assembly=None, save=save, **kwargs)
        pf = PDBFrame._wrap(pf.df, pf.name, {**(pf.metadata or {}), 'assembly': assembly},
                            pf._seqres)
        return annotate(pf, complex=complex)

    @classmethod
    def from_biopython(cls, structure, model: int = 0, hetero: bool = False,
                       name: str | None = None) -> TCRFrame:
        """A Bio.PDB Structure, annotated; see :meth:`PDBFrame.from_biopython`."""
        return annotate(PDBFrame.from_biopython(structure, model=model, hetero=hetero,
                                                name=name))

    @classmethod
    def from_sequences(cls, sequences, name: str | None = None) -> TCRFrame:
        """Sequences only, annotated; see :meth:`PDBFrame.from_sequences`."""
        return annotate(PDBFrame.from_sequences(sequences, name=name))

    @classmethod
    def ensure(cls, data) -> TCRFrame:
        """`data` as a TCRFrame: a TCRFrame, a PDBFrame, a file path, an atom table or a
        Bio.PDB structure (annotated if needed)."""
        return annotate(data)

    def _same(self, df: pd.DataFrame, seqres=False) -> TCRFrame:
        frame = super()._same(df, seqres)
        frame._parent = (self, None)
        return frame

    def _subset(self, rows, unobserved: np.ndarray | None = None) -> TCRFrame:
        frame = super()._subset(rows, unobserved)
        frame._parent = (self, np.asarray(rows))
        return frame

    # ---------------------------------------------------------------- chains

    @cached_property
    def chain_table(self) -> pd.DataFrame:
        """One row per chain: ``role``, ``gene``, ``allele``, ``species``, ``identity``,
        ``j_gene``, ``j_identity``, ``original_chain``, ``length``, ``sequence``.

        MHC and B2M chains: the gene, its closest allele, that allele's species and the
        identity to it over the aligned part (:func:`.mhc.mhc_match`). TCR chains: the V
        gene (``'TRAV12-2'``), its closest allele (``'TRAV12-2*01'``), the germline's
        species and ANARCI's identity to it; then the J allele and its identity. Other
        chains: ``'-'`` and NaN.
        """
        df = self.df
        rows = []
        for chain, seq in self.sequences.items():
            role = _ROLES.get(chain, 'other')
            if chain == 'A' and self.mhc_class == 'II':
                role = 'MHC alpha'
            gene = allele = j_gene = species = '-'
            identity = j_identity = np.nan
            if chain in ('A', 'B', 'F'):
                match = _mhc.mhc_match(seq)
                gene, allele = (x if isinstance(x, str) and x else '-'
                                for x in (match['gene'], match['allele']))
                if gene != '-':
                    identity = float(match['allele_identity'])
                    species = mhc_species(allele) or '-'
            elif chain in ('D', 'E'):
                allele, identity, j_gene, j_identity, species = _germline(seq)
                gene = allele.split('*')[0]
            original = (df.loc[df['chain'] == chain, 'original_chain'].iloc[0]
                        if 'original_chain' in df else chain)
            rows.append({'chain': chain, 'role': role, 'gene': gene, 'allele': allele,
                         'species': species, 'identity': identity, 'j_gene': j_gene, 'j_identity': j_identity,
                         'original_chain': original, 'length': len(seq), 'sequence': seq})
        return pd.DataFrame(rows).set_index('chain')

    @property
    def mhc_class(self) -> str | None:
        """``'I'``, ``'II'`` (an MHC beta chain B is present), or None without an MHC."""
        chains = self.chains
        return 'II' if 'B' in chains else 'I' if 'A' in chains else None

    @property
    def has_tcr(self) -> bool:
        return {'D', 'E'} <= set(self.chains)

    @property
    def kind(self) -> str | None:
        """What the structure is, from its chains: ``'tcrpmhc'`` (TCR, MHC and peptide),
        ``'pmhc'`` (MHC and peptide), ``'mhc'``, ``'tcr'`` (TCR alpha and beta only), or
        None for anything else."""
        chains = set(self.chains)
        mhc, peptide = 'A' in chains, 'C' in chains
        if self.has_tcr:
            return 'tcrpmhc' if mhc and peptide else 'tcr' if not (mhc or peptide) else None
        if mhc:
            return 'pmhc' if peptide else 'mhc'
        return None

    @property
    def mhc_species(self) -> str | None:
        """The MHC's species: that of chain A's closest MHC sequence (B's, then the
        B2M's, without an A), e.g. ``'human'``, ``'mouse'``, ``'pig'``; None if unknown."""
        return self._species_of('A', 'B', 'F')

    @property
    def tcr_species(self) -> str | None:
        """The TCR's species: that of the alpha chain's germline genes (the beta's,
        without an alpha), ``'human'`` or ``'mouse'``, the TCR germlines ANARCI has;
        None without a TCR."""
        return self._species_of('D', 'E')

    @property
    def species(self) -> str | None:
        """The complex's species: the TCR's (:attr:`tcr_species`), or the MHC's for a
        structure without a TCR."""
        return self.tcr_species or self.mhc_species

    def _species_of(self, *chains: str) -> str | None:
        ch = self.chain_table
        for c in chains:
            if c in ch.index and ch.at[c, 'species'] != '-':
                return ch.at[c, 'species']
        return None

    @cached_property
    def residues(self) -> pd.DataFrame:
        """The residue table with each residue's ``imgt`` label, ``original_chain`` and
        ``region``.

        ``region`` is the most specific region a residue is in: a CDR, the
        peptide core, the peptide, the MHC helices, a V domain, or ``''``. Read from
        mmCIF, the residues without atoms are there too (``observed`` False; see
        PDBFrame's ``residues``), with their IMGT label (the chain is numbered on its
        deposited sequence) and, from it, their CDR, V domain or peptide.
        """
        df = self.df
        keys = ['chain', 'observed_index']
        res = PDBFrame.residues.fget(self)
        extra = [c for c in ('imgt', 'original_chain') if c in df]
        res = res.merge(df.drop_duplicates(keys)[keys + extra].astype({'observed_index': 'Int64'}),
                        on=keys, how='left')
        label = np.full(len(df), '', dtype=object)
        # least specific first, so the more specific regions overwrite
        order = ['Va', 'Vb', 'helices', 'peptide', 'peptide_core', *CDRS]
        for name in order:
            if name in self.masks:
                label[self.masks[name]] = name
        keyed = pd.Series(label, index=pd.MultiIndex.from_frame(df[keys]))
        keyed = keyed[~keyed.index.duplicated()]
        seen = res['observed'].to_numpy()
        region = np.full(len(res), '', dtype=object)
        region[seen] = keyed.reindex(pd.MultiIndex.from_frame(
            res.loc[seen, keys].astype({'observed_index': int}))).to_numpy()
        res['region'] = region
        if seen.all():
            return res
        # residues without atoms: labels from the deposited sequence
        seqres = self._seqres
        at = pd.MultiIndex.from_frame(res.loc[~seen, ['chain', 'full_index']].astype(
            {'full_index': int}))
        index = pd.MultiIndex.from_frame(seqres[['chain', 'full_index']])
        if 'imgt' in res:
            imgt = seqres['imgt'] if 'imgt' in seqres else pd.Series('-', index=seqres.index)
            res.loc[~seen, 'imgt'] = pd.Series(imgt.to_numpy(), index=index).reindex(at).to_numpy()
        if 'original_chain' in res:
            files = df.drop_duplicates('chain').set_index('chain')['original_chain']
            res.loc[~seen, 'original_chain'] = res.loc[~seen, 'chain'].map(files).to_numpy()
        unseen = np.full(len(seqres), '', dtype=object)
        masks = _regions.unobserved_regions(self)
        for name in order:
            if name in masks and name in self.masks:
                unseen[masks[name]] = name
        res.loc[~seen, 'region'] = pd.Series(unseen, index=index).reindex(at).to_numpy()
        return res

    # --------------------------------------------------------------- regions

    @cached_property
    def masks(self) -> dict[str, np.ndarray]:
        """``{region: boolean atom mask}`` for every region that could be found, by its
        canonical name (see :data:`REGIONS`; :meth:`describe` says what each one is).

        A frame derived from another (a selection, a transformed copy, ...) keeps
        the regions of the structure it came from: ``pf.select(chains='D').masks``
        has chain D's CDRs as found in the whole complex.
        """
        parent = self.__dict__.get('_parent')
        if parent is not None:
            frame, rows = parent
            if rows is None:
                return frame.masks
            sliced = {name: m[rows] for name, m in frame.masks.items()}
            return {name: m for name, m in sliced.items() if m.any()}
        out = {}

        def add(name, fn):
            try:
                mask = fn()
            except Exception as e:
                warn(f'{self.name}: region {name!r} not found: {e}', stacklevel=5)
                return
            if mask.any():
                out[name] = mask

        chains = set(self.chains)
        if 'C' in chains:
            add('peptide', lambda: (self.df['chain'] == 'C').to_numpy())
            add('peptide_core', lambda: _regions.peptide_core(self))
            # a region by distance: none without coordinates (TCRFrame.from_sequences)
            if 'peptide_core' in out and self.has_coordinates:
                add('peptide_centric',
                    lambda: _regions.near_residues(self, out['peptide_core'], 10.0))
        for c in CDRS:
            # each CDR needs only its own chain, so a one-chain selection keeps its CDRs
            if _regions._TCR_CHAIN[c[3]] in chains:
                add(c, lambda c=c: _regions.cdr(self, c))
        if 'A' in chains:
            add('helices', lambda: _mhc.mhc_helices(self))
        for name, chain in [('Va', 'D'), ('Vb', 'E')]:
            if chain in chains:
                add(name, lambda chain=chain: _regions.v_domain(self, chain))
        for name, ids in _CHAIN_REGIONS.items():
            if set(ids) & chains:
                out[name] = self.df['chain'].isin(ids).to_numpy()
        return out

    @cached_property
    def regions(self) -> dict[str, TCRFrame]:
        """``{region: TCRFrame}``: each region as its own structure."""
        return {name: PDBFrame.select(self, mask=m) for name, m in self.masks.items()}

    def select(self, chains=None, residues=None, atoms=None, mask=None,
               invert: bool = False, regions=None) -> TCRFrame:
        """Keep the atoms matching every filter given; see :meth:`PDBFrame.select`.

            >>> pf.select(regions='cdra1')                # one CDR
            >>> pf.select(regions=['cdr3', 'peptide_core'])   # CDR3 alpha and beta, and the core
            >>> pf.select(regions='tcr', atoms='ca')      # CA atoms of both TCR chains

        Args:
            regions: one or more regions, by name or group, in any case (see
                :data:`REGIONS`, :data:`REGION_GROUPS` and :meth:`describe`). A
                group skips members this structure lacks (``'pmhc'`` has no B2M in
                class II); a missing region named on its own is an error.
            chains, residues, atoms, mask, invert: as in :meth:`PDBFrame.select`.
        """
        if regions is None:
            return super().select(chains=chains, residues=residues, atoms=atoms, mask=mask,
                                  invert=invert)
        names = self._region_names(regions)
        picked = np.logical_or.reduce([self.masks[n] for n in names])
        if self._seqres is None or mask is not None or (invert and atoms is not None):
            mask = picked if mask is None else np.asarray(mask) & picked
            return super().select(chains=chains, residues=residues, atoms=atoms, mask=mask,
                                  invert=invert)
        # residues without atoms: in a CDR, V domain or the peptide by their labels, in
        # other regions (peptide core, helices, ...) when inside what is kept
        keep = select_mask(self.df, chains=chains, residues=residues, atoms=atoms, mask=picked)
        by_label = _regions.unobserved_regions(self)
        by_label.update({n: self._seqres['chain'].isin(ids).to_numpy()
                         for n, ids in _CHAIN_REGIONS.items()})
        unseen = np.logical_or.reduce(
            [by_label[n] if n in by_label else deposited.inside(self._seqres, self.df[picked])
             for n in names])
        unseen &= select_mask(self._seqres, chains=chains, residues=residues)
        if invert:
            keep, unseen = ~keep, ~unseen
        return self._subset(keep, unseen)

    def _region_names(self, regions) -> list[str]:
        """The regions of :meth:`select`'s `regions` this structure has."""
        items = [regions] if isinstance(regions, str) else list(regions)
        names, missing = [], []
        for item in items:
            members = expand_regions(item)
            grouped = _key(item) in _GROUP_LOOKUP
            names += [n for n in members if n in self.masks]
            missing += [] if grouped else [n for n in members if n not in self.masks]
        if missing or not names:
            raise KeyError(f'{self.name} has no {missing or items}; it has {list(self.masks)}')
        return list(dict.fromkeys(names))

    @staticmethod
    def describe() -> pd.DataFrame:
        """What each region and group is."""
        rows = [{'name': n, 'kind': 'region', 'what': d} for n, d in _DESCRIPTIONS.items()]
        rows += [{'name': g, 'kind': 'group', 'what': ', '.join(m)}
                 for g, m in REGION_GROUPS.items()]
        return pd.DataFrame(rows).set_index('name')

    @property
    def peptide(self) -> str | None:
        return self.sequences.get('C')

    @property
    def peptide_core(self) -> TCRFrame | None:
        """The peptide core (region ``peptide_core``) as a TCRFrame."""
        return self.regions.get('peptide_core')

    @property
    def peptide_core_sequence(self) -> str | None:
        """The peptide core's sequence; the same as ``s.sequence('peptide_core')``."""
        return self.sequence('peptide_core')

    def sequence(self, name: str) -> str | None:
        """The sequence of a chain or of a region that lies in one chain.

            >>> s.sequence('D'), s.sequence('peptide_core'), s.sequence('cdrb3')

        Args:
            name: a chain ID (``'A'`` ... ``'F'``) or a region name (any case, as for
                :meth:`select`).

        Returns:
            The one-letter sequence, or None if the structure lacks that chain or
            region. For a region over several chains (``'peptide_centric'``, a
            group), use ``s.select(regions=name).sequences``.
        """
        if name in self.chains:
            return self.sequences[name]
        if len(name) == 1 and name.isupper():
            return None
        regions = expand_regions(name)
        if len(regions) != 1:
            raise ValueError(f'{name!r} is a group of {len(regions)} regions; use '
                             f"s.select(regions={name!r}).sequences")
        region = self.regions.get(regions[0])
        if region is None:
            return None
        seqs = region.sequences
        if len(seqs) != 1:
            raise ValueError(f'region {regions[0]} spans chains {", ".join(seqs)}; use '
                             f"s.select(regions={name!r}).sequences")
        return next(iter(seqs.values()))

    @cached_property
    def cdrs(self) -> dict[str, str]:
        """``{CDR: sequence}`` for the CDRs found."""
        return {c: self.sequence(c) for c in CDRS if c in self.regions}

    # ---------------------------------------------------------------- angles

    @cached_property
    def angles(self) -> dict[str, float]:
        """Crossing and incident docking angles (degrees, ``method='piercelab'``); NaN,
        with a warning, if not computable. See :meth:`docking_angles` for the other method."""
        return self.docking_angles()

    def docking_angles(self, method: str | dict = 'piercelab', *, helices: str | None = None,
                       axis: str | None = None, signed: bool | None = None) -> dict[str, float]:
        """Crossing and incident docking angles (degrees) by `method`.

        Args:
            method: ``'piercelab'`` (default; Pierce's helix residues, V-domain
                rotation axis, 0-180) or ``'v1'`` (Motif
                Atlas helices, whole-chain axis, 0-90).
            helices, axis, signed: override one part of the method; see
                :func:`.angles.docking_angles`.
        """
        settings = angle_settings(method, helices=helices, axis=axis, signed=signed)
        try:
            return docking_angles(self, settings)
        except Exception as e:
            warn(f'{self.name}: docking angles not computed: {e}', stacklevel=5)
            return {'crossing_angle': np.nan, 'incident_angle': np.nan}

    def docking_vectors(self, method: str | dict = 'piercelab', **parts) -> dict:
        """The vectors behind :meth:`docking_angles`; see :func:`.angles.docking_vectors`."""
        return docking_vectors(self, method, **parts)

    # --------------------------------------------------------------- summary

    def summary(self) -> pd.Series:
        """The main facts in one Series."""
        ch = self.chain_table
        genes = {r: ch.at[c, 'allele'] for c, r in [('A', 'mhc'), ('B', 'mhc_beta'),
                                                    ('D', 'TRAV'), ('E', 'TRBV')]
                 if c in ch.index}
        return pd.Series({
            'name': self.name, 'mhc_class': self.mhc_class, 'species': self.species,
            'tcr_species': self.tcr_species, 'mhc_species': self.mhc_species, **genes,
            'peptide': self.peptide, 'peptide_core': self.peptide_core_sequence,
            **self.cdrs, **(self.angles if self.has_tcr else {}),
            'chains': ', '.join(f'{c}={r}' for c, r in ch['role'].items()),
        }, name=self.name)

    def _summary(self) -> str:
        """One line: the name, MHC class and chains (nothing that needs computing)."""
        mhc = f', MHC class {self.mhc_class}' if self.mhc_class else ''
        return f"TCRFrame {self.name!r}{mhc}, chains {', '.join(self.chains)}"

    def __repr__(self) -> str:
        return self._summary()

    def _repr_html_(self) -> str:
        return f'<pre>{html.escape(self._summary())}</pre>'

    def report(self) -> str:
        """A few lines on the complex: each chain's role, gene and length, the peptide and
        its core, and the CDRs (what ``tcrstruct info`` prints)."""
        ch = self.chain_table
        lines = [self._summary()]
        if self.species:
            parts = [f'{k} {v}' for k, v in (('TCR', self.tcr_species),
                                             ('MHC', self.mhc_species)) if v]
            lines.append(f'  species {self.species} ({", ".join(parts)})')
        for chain, r in ch.iterrows():
            gene = '' if r['allele'] in ('-', None) else r['allele']
            lines.append(f'  {chain}  {r["role"]:<16}{gene:<12}{r["length"]:>4} aa'
                         f'   (chain {r["original_chain"]} in the file)')
        if self.peptide:
            core = self.peptide_core_sequence
            lines.append(f'  peptide {self.peptide}' + (f'   core {core}' if core and
                                                         core != self.peptide else ''))
        if self.cdrs:
            lines.append('  CDRs    ' + '  '.join(f'{k}={v}' for k, v in self.cdrs.items()))
        return '\n'.join(lines)

    # --------------------------------------------------------------- actions

    def cdr_contacts(self, cutoff: float = 4.5) -> pd.DataFrame:
        """Per CDR: closest distances to the peptide and the MHC, and its residues
        touching each; see :func:`.contacts.cdr_contacts`."""
        from .contacts import cdr_contacts
        return cdr_contacts(self, cutoff)

    def contact_map(self, rows='cdrs', columns='peptide', cutoff: float = 5.0,
                    grid: bool = False) -> pd.DataFrame:
        """Residue pairs of two regions within `cutoff` Å; see :func:`.contacts.contact_map`."""
        from .contacts import contact_map
        return contact_map(self, rows, columns, cutoff, grid)

    def split(self, parts=None, out_dir: str | Path | None = None, fmt: str = 'pdb') -> dict:
        """The TCR, pMHC, MHC and peptide (or other `parts`) as their own structures, or
        files in `out_dir`; see :func:`.split.split`."""
        from .split import SPLIT_PARTS, split
        return split(self, tuple(SPLIT_PARTS) if parts is None else parts, out_dir, fmt)

    def view(self, regions=True, **kwargs):
        """3D view with regions marked; see :func:`.view.view`."""
        from .view import view
        return view(self, regions=regions, **kwargs)

    def tcr_up(self) -> TCRFrame:
        """A copy whose coordinates are turned TCR up, peptide left to right, MHC below
        (see :func:`.angles.tcr_up_orientation`); ``view(tcr_up=True)`` shows the
        same without changing the coordinates."""
        from .angles import tcr_up_orientation
        return self.transform(*tcr_up_orientation(self))

    def compare(self, reference, **kwargs) -> pd.Series:
        """Benchmark metrics of this structure (as the model) against `reference`;
        see :func:`.compare.compare`."""
        from .compare import compare
        return compare(self, reference, **kwargs)

    def normalize(self, **kwargs) -> TCRFrame:
        """The structure with TCRs numbered by IMGT; see :func:`.chains.normalize`."""
        return normalize(self, **kwargs)

    def verify(self, **kwargs) -> pd.DataFrame | bool:
        """Is it a complete TCR-pMHC (or pMHC with ``kind='pmhc'``)? One row per check,
        or just True / False with ``as_bool=True``; see :func:`.verify.verify`."""
        from .verify import verify
        return verify(self, **kwargs)

    def is_valid(self, **kwargs) -> bool:
        """Whether :meth:`verify` finds nothing failing (``verify(as_bool=True)``)."""
        return self.verify(as_bool=True, **kwargs)

    def find_fused_peptide(self, **kwargs) -> pd.DataFrame:
        """A peptide fused to another chain, if any; see :func:`.fused.find_fused_peptide`."""
        from .fused import find_fused_peptide
        return find_fused_peptide(self, **kwargs)

    def split_fused_peptide(self, **kwargs) -> TCRFrame:
        """The structure with a fused peptide moved into chain C; see
        :func:`.fused.split_fused_peptide`."""
        from .fused import split_fused_peptide
        return split_fused_peptide(self, **kwargs)
