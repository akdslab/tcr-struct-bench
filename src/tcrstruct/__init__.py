"""tcrstruct - TCR-pMHC structures on top of PDBFrame: annotation, regions, benchmarking.

    >>> import tcrstruct as tcs
    >>> from tcrstruct import TCRFrame
    >>> pdb1 = TCRFrame.read('7n2n.pdb')      # a PDBFrame with chains A-F and IMGT labels
    >>> pdb1 = TCRFrame.from_rcsb('7N2N')     # the same, from the RCSB PDB
    >>> pdb1                                  # name, MHC class, chains
    >>> print(pdb1.report())                  # + genes, peptide core, CDRs
    >>> pdb1.residues                         # residue table with 'imgt' and 'region'
    >>> tcs.list_regions()                    # every region, and how it is defined
    >>> pdb1.view()                           # 3D with the regions marked
    >>> tcs.compare(model, pdb1)              # one row of benchmark metrics
    >>> tcs.compare(model, pdb1, **tcs.V1)    # ... the 'v1' versions of the metrics
    >>> tcs.compare_dirs('models/', 'refs/')  # one row per model
    >>> tcs.normalize(pdb1).write('7n2n_imgt.pdb')
    >>> tcs.find_complexes('2e7l.cif')         # each copy of the complex in a file
    >>> pdb1.split(out_dir='parts/')          # TCR, pMHC, MHC, peptide files
    >>> tcs.docs('metrics')                   # the documentation (tcrstruct docs on the CLI)
    >>> tcs.install_skills()                  # the Claude Code skills, into ~/.claude/skills

A :class:`TCRFrame` is a PDBFrame, always annotated (see :mod:`.tcr`).

    numbering.py   ANARCI IMGT numbering and per-residue IMGT labels
    chains.py      chain detection (TCR / MHC / B2M / peptide), annotate, normalize
    examples.py    the example structures (tests/data): tcs.example('1AO7', 'rcsb.pdb')
    complexes.py   several complexes in one file (asymmetric unit): find each by contacts
    regions.py     peptide core, peptide-centric region, CDRs (atom masks)
    mhc.py         IMGT G-domain numbering of the MHC chains, and the MHC groove helices
    tcr.py         TCRFrame: the annotated PDBFrame, with regions, CDRs, angles
    angles.py      crossing and incident docking angles
    contacts.py    contact maps between regions (CDRs x peptide, ...)
    metrics.py     matching, superposition and every score
    compare.py     compare() and the reader for prepared structures (*_processed.csv)
    fused.py       peptides fused to another chain: find and split them
    split.py       a complex split into TCR, pMHC, MHC, peptide (or any region), as files
    verify.py      is it a complete TCR-pMHC / pMHC: chains, contacts, breaks, anchors
    sources.py     structure lists (TCR3d, STCRDab), RCSB metadata, batch RCSB downloads
    batch.py       directories: pairing files by PDB ID, compare_dirs, normalize_dir
    view.py        3D views with regions marked, model-on-reference comparison
    resources.py   the documentation and the Claude Code skills that come with it
    cli.py         the `tcrstruct` command
"""

from . import mhc, sources
from ._warnings import TCRStructWarning
from .batch import compare_dirs, compare_models, compare_pairs, normalize_dir, pair_files
from .chains import annotate, detect_chains, normalize
from .complexes import complex_table, find_complexes
from .contacts import cdr_contacts, contact_map, plot_contact_map
from .examples import EXAMPLE_DATA, example
from .compare import (COLUMNS, METRIC_GROUPS, NEW_COLUMNS, SETTINGS, V1, compare,
                      group_columns, read_processed, versions)
from .metrics import per_residue_lddt
from .resources import docs, install_skills
from .fused import find_fused_peptide, split_fused_peptide
from .split import split
from .verify import is_valid, verify, verify_dir
from .tcr import REGION_GROUPS, REGIONS, TCRFrame, list_region_groups, list_regions
from .view import view, view_comparison

__all__ = ['TCRFrame', 'annotate', 'normalize', 'detect_chains', 'REGIONS',
           'REGION_GROUPS', 'list_regions', 'list_region_groups', 'compare',
           'compare_dirs', 'compare_pairs', 'compare_models', 'pair_files', 'normalize_dir',
           'read_processed', 'view', 'view_comparison', 'COLUMNS', 'NEW_COLUMNS', 'SETTINGS',
           'METRIC_GROUPS', 'V1', 'versions', 'group_columns', 'per_residue_lddt',
           'contact_map', 'example', 'EXAMPLE_DATA',
           'plot_contact_map', 'cdr_contacts', 'find_complexes', 'complex_table', 'split',
           'TCRStructWarning', 'find_fused_peptide',
           'split_fused_peptide', 'verify', 'verify_dir', 'is_valid', 'docs', 'install_skills']
__version__ = '0.2.2'
