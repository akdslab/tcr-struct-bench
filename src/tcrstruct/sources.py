"""Lists of TCR-pMHC and pMHC structures, and their files, from public databases.

    >>> tcs.sources.tcr3d_complexes()                 # TCR-pMHC: PDB ID, class, genes, CDRs, ...
    >>> tcs.sources.tcr3d_pmhc('II')                  # class II pMHC structures
    >>> tcs.sources.stcrdab()                         # every TCR structure in STCRDab
    >>> tcs.sources.rcsb_info(['1AO7', '6XCP'])       # title, release date, resolution, ...
    >>> tcs.sources.rcsb_download(['1AO7', '6XCP'], 'structures/', n_jobs=4)
    >>> TCRFrame.from_rcsb('6XCP')                    # one entry, straight into a TCRFrame

Sources:

- **TCR3d** (https://tcr3d.ibbr.umd.edu):
  - the TCR-pMHC table is its ``tcr_complexes_data.tsv`` download;
  - the pMHC tables are read from the JSON that its *MHC class I / II chains* pages embed.
- **STCRDab** (https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred): its summary TSV.
- **RCSB PDB**: entry metadata and files (mmCIF, PDB, FASTA, biological assemblies)
  come through PDBFrame's ``pdbframe.structure.rcsb``; tcrstruct runs the downloads
  over many IDs at once in :func:`rcsb_download`.

Only the standard library is used to download (``urllib``), so no extra packages are
needed. Results are not cached; save the tables you use, e.g.
``table.to_csv(...)``, so a benchmark set stays fixed.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

__all__ = ['tcr3d_complexes', 'tcr3d_pmhc', 'stcrdab', 'rcsb_info', 'rcsb_download']

TCR3D = 'https://tcr3d.ibbr.umd.edu'
STCRDAB = 'https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred'

_CLASS = {'I': 'CLASSI', 'II': 'CLASSII', 1: 'CLASSI', 2: 'CLASSII'}


def _get(url: str, timeout: float = 60) -> bytes:
    """The bytes at `url`, with an error that says which URL failed."""
    req = urllib.request.Request(url, headers={'User-Agent': 'tcrstruct'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise ConnectionError(f'{url}: HTTP {e.code} {e.reason}') from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ConnectionError(f'{url}: {getattr(e, "reason", e)} (no network?)') from e


def _class_filter(df: pd.DataFrame, column: str, mhc_class) -> pd.DataFrame:
    if mhc_class is None:
        return df.reset_index(drop=True)
    if mhc_class not in _CLASS:
        raise ValueError(f"mhc_class must be 'I', 'II' or None, not {mhc_class!r}")
    return df[df[column] == _CLASS[mhc_class]].reset_index(drop=True)


# ----------------------------------------------------------------------- TCR3d

def tcr3d_complexes(mhc_class=None) -> pd.DataFrame:
    """TCR3d's TCR-pMHC complexes: one row per PDB entry.

    Columns include ``pdb_id`` (upper case), ``mhc_class`` (``'I'`` / ``'II'``),
    ``TCR_name``, ``Epitope``, ``MHC_allele``, the V/J genes, the six CDR
    sequences, TCR3d's docking and incident angles, ``Release_date`` and
    ``Resolution``.

    Args:
        mhc_class: ``'I'``, ``'II'`` or None for both.
    """
    raw = _get(f'{TCR3D}/static/download/tcr_complexes_data.tsv')
    df = pd.read_csv(io.BytesIO(raw), sep='\t')
    df = _class_filter(df, 'TCR_complex', mhc_class)
    df.insert(0, 'pdb_id', df.pop('PDB_ID').str.upper())
    df.insert(1, 'mhc_class', df['TCR_complex'].map({'CLASSI': 'I', 'CLASSII': 'II'}))
    df['Release_date'] = pd.to_datetime(df['Release_date'], errors='coerce')
    return df


def tcr3d_pmhc(mhc_class=None) -> pd.DataFrame:
    """TCR3d's peptide-MHC structures (with or without a TCR): one row per PDB entry.

    Columns: ``pdb_id``, ``mhc_class``, ``mhc_allele_name``, ``mhc_species``,
    ``peptide``, ``is_bound`` (1 when the entry also has a TCR), ``release_date``,
    ``resolution``, ``pmid``.

    Args:
        mhc_class: ``'I'``, ``'II'`` or None for both.
    """
    pages = {'I': 'mhc1_chains', 'II': 'mhc2_chains'}
    wanted = list(pages) if mhc_class is None else [{1: 'I', 2: 'II'}.get(mhc_class, mhc_class)]
    frames = []
    for c in wanted:
        if c not in pages:
            raise ValueError(f"mhc_class must be 'I', 'II' or None, not {mhc_class!r}")
        html = _get(f'{TCR3D}/{pages[c]}').decode()
        m = re.search(r'var data = (\[.*?\]);', html, re.S)
        if m is None:
            raise ValueError(f'{TCR3D}/{pages[c]}: table data not found; the page layout '
                             'may have changed')
        frames.append(pd.DataFrame(json.loads(m.group(1))))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, 'pdb_id', df.pop('pdbid').str.upper())
    df.insert(1, 'mhc_class', df.pop('mhc_class').map({'CLASSI': 'I', 'CLASSII': 'II'}))
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    return df


# --------------------------------------------------------------------- STCRDab

def stcrdab(peptide_mhc_only: bool = True) -> pd.DataFrame:
    """STCRDab's summary of TCR structures: one row per TCR in each entry.

    It names the chains of each TCR, antigen and MHC in the file (``Achain``,
    ``Bchain``, ``antigen_chain``, ``mhc_chain1`` / ``mhc_chain2``), with species,
    resolution and release date.

    Args:
        peptide_mhc_only: keep alpha/beta TCRs bound to a peptide on an MHC (drops
            CD1 / MR1, gamma-delta TCRs and TCRs without antigen).
    """
    raw = _get(f'{STCRDAB}/summary/all')
    df = pd.read_csv(io.BytesIO(raw), sep='\t', dtype=str, keep_default_na=False)
    if peptide_mhc_only:
        df = df[(df['TCRtype'] == 'abTCR') & (df['antigen_type'] == 'peptide')
                & df['mhc_type'].isin(['MH1', 'MH2', 'GA', 'GB', 'GA1', 'GA2'])]
    df = df.reset_index(drop=True)
    df.insert(0, 'pdb_id', df.pop('pdb').str.upper())
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
    return df


# ------------------------------------------------------------------------ RCSB

def rcsb_info(pdb_ids) -> pd.DataFrame:
    """Entry metadata from RCSB: ``title``, ``release_date``, ``method``,
    ``resolution``, ``n_entities`` per PDB ID (one request each). The same as
    PDBFrame's :meth:`PDBFrame.rcsb_info`."""
    from pdbframe import PDBFrame
    return PDBFrame.rcsb_info(pdb_ids)


def _download_one(args) -> dict:
    from pdbframe.structure import rcsb
    pid, out_dir, fmt, assembly, overwrite = args
    if fmt == 'fasta':
        assembly = None                       # sequences: one per entry

    def target(n):
        path = Path(out_dir) / rcsb.url(pid, n, fmt).rsplit('/', 1)[1]
        return path.with_name(path.name + '.fasta') if fmt == 'fasta' else path

    path = target(assembly)
    if path.exists() and not overwrite:
        return {'pdb_id': pid, 'file': str(path), 'status': 'exists'}
    try:
        rcsb.fetch(pid, assembly=assembly, fmt=fmt, save=path)
        return {'pdb_id': pid, 'file': str(path), 'status': 'downloaded'}
    except Exception as e:
        if assembly is None or 'HTTP 404' not in str(e):
            return {'pdb_id': pid, 'file': '', 'status': f'failed: {e}'}
    path = target(None)                       # no such assembly: the deposited structure
    try:
        rcsb.fetch(pid, assembly=None, fmt=fmt, save=path)
        return {'pdb_id': pid, 'file': str(path),
                'status': f'downloaded (no assembly {assembly}: deposited structure)'}
    except Exception as e:
        return {'pdb_id': pid, 'file': '', 'status': f'failed: {e}'}


def rcsb_download(pdb_ids, out_dir: str | Path, fmt: str = 'cif', assembly: int | None = 1,
                  overwrite: bool = False, n_jobs: int = 4, progress: bool = False
                  ) -> pd.DataFrame:
    """Download many RCSB entries into `out_dir`, in parallel.

    Each file is fetched with PDBFrame's ``pdbframe.structure.rcsb.fetch`` (the
    same as ``PDBFrame.from_rcsb(..., save=...)``) and keeps RCSB's file name:
    ``1AO7.cif``, ``1AO7-assembly1.cif``, ``1AO7.pdb``, ``1AO7.fasta``.

    Args:
        pdb_ids: PDB IDs (any case), e.g. ``tcr3d_complexes()['pdb_id']``.
        fmt: ``'cif'`` (every entry has one), ``'pdb'`` (missing for large entries)
            or ``'fasta'`` (the full deposited sequences).
        assembly: the biological assembly (default 1, as :meth:`TCRFrame.from_rcsb`;
            an entry without it gets its deposited structure); ``None``: the deposited
            structure. Not used for FASTA.
        overwrite: download again files that are already there.
        n_jobs: parallel downloads.

    Returns:
        ``pdb_id``, ``file`` and ``status`` (``'downloaded'``, ``'exists'`` or
        ``'failed: ...'``) per ID. Failures are listed, with a warning, not raised;
        so are entries without the assembly.
    """
    from ._warnings import warn
    from .batch import _map
    if fmt not in ('cif', 'pdb', 'fasta'):
        raise ValueError(f"fmt must be 'cif', 'pdb' or 'fasta', not {fmt!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(dict.fromkeys(p.strip().upper() for p in ([pdb_ids] if isinstance(pdb_ids, str)
                                                          else pdb_ids)))
    rows = _map(_download_one, [(p, out_dir, fmt, assembly, overwrite) for p in ids], n_jobs,
                progress, 'downloading')
    table = pd.DataFrame(rows, columns=['pdb_id', 'file', 'status'])
    fallback = table[table['status'].str.contains('no assembly')]
    if len(fallback):
        warn(f'{len(fallback)} entries have no biological assembly {assembly}, so their '
             'deposited structure was downloaded: ' + ', '.join(fallback['pdb_id'][:10])
             + (' ...' if len(fallback) > 10 else ''))
    failed = table[table['status'].str.startswith('failed')]
    if len(failed):
        warn(f'{len(failed)} of {len(table)} RCSB downloads failed: '
             + ', '.join(failed['pdb_id'][:10]) + (' ...' if len(failed) > 10 else ''))
    return table
