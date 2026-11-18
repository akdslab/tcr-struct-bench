"""Many structures at once: score a directory of models, normalize a directory.

    >>> table = compare_dirs('predictions/', 'references/', n_jobs=8)
    >>> table.groupby('method')['DockQ'].mean()
    >>> normalize_dir('raw/', 'normalized/')

Models are paired with references by PDB ID: the first 4 characters of the file
name, upper-cased. So ``7N2N_model_0.pdb``,
``7n2n_seed2.cif`` and ``7N2N.pdb`` in the model directory are all scored against
``7N2N.pdb`` (or ``7n2n.cif``) in the reference directory. Pass ``key=`` for
another rule.

Directories are searched recursively for ``.pdb``, ``.cif``, ``.ent``, ``.mmcif``
(and ``.gz`` of each); files with ``_processed`` in the name are skipped.
"""

from __future__ import annotations

import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from ._warnings import warn
from .chains import annotate, normalize
from .compare import (COLUMNS, NEW_COLUMNS, SETTINGS, _modes, _settings, compare, group_columns,
                      versions)

__all__ = ['find_structures', 'pdb_key', 'key_pattern', 'pair_files', 'compare_pairs',
           'compare_models',
           'compare_dirs', 'normalize_dir', 'STRUCTURE_SUFFIXES']

STRUCTURE_SUFFIXES = ('.pdb', '.cif', '.ent', '.mmcif')


def pdb_key(path: str | Path) -> str:
    """The PDB ID a file belongs to: the first 4 characters of its name, upper-cased."""
    return Path(path).name[:4].upper()


def _stem(path: Path) -> str:
    name = path.name[:-3] if path.name.endswith('.gz') else path.name
    return name.rsplit('.', 1)[0]


def find_structures(directory: str | Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Structure files under `directory` (recursively), sorted.

    Args:
        exclude: skip files inside sub-directories with these names (e.g. ``('rcsb',)``).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f'not a directory: {directory}')
    files = []
    for p in directory.rglob('*'):
        name = p.name[:-3] if p.name.endswith('.gz') else p.name
        if (p.is_file() and name.lower().endswith(STRUCTURE_SUFFIXES)
                and '_processed' not in name
                and not set(p.relative_to(directory).parts[:-1]) & set(exclude)):
            files.append(p)
    return sorted(files)


def pair_files(models: list[Path], references: list[Path],
               key: Callable[[Path], str] = pdb_key) -> tuple[pd.DataFrame, list[Path]]:
    """Pair every model with the reference of the same key.

    Returns:
        ``(pairs, unmatched)``: a table with ``PDB``, ``model_file``,
        ``reference_file``, and the models without a reference.

    Raises:
        ValueError: when two references share a key, since the pairing would be
            ambiguous.
    """
    refs: dict[str, Path] = {}
    for r in references:
        k = key(r)
        if k in refs:
            raise ValueError(f'references {refs[k].name} and {r.name} both have key {k!r}; '
                             'keep one, or pass key= to tell them apart')
        refs[k] = r
    rows, unmatched = [], []
    for m in models:
        k = key(m)
        if k in refs:
            rows.append({'PDB': k, 'model_file': m, 'reference_file': refs[k]})
        else:
            unmatched.append(m)
    return pd.DataFrame(rows, columns=['PDB', 'model_file', 'reference_file']), unmatched


# ------------------------------------------------------------------- scoring

def _recorded(fn, *args, **kwargs) -> tuple[object, list[str]]:
    """``fn(...)`` and the messages of the warnings it raised.

    Warnings raised in worker processes never reach the caller, so they travel
    back with the result and are raised again there (see :func:`_replay`).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn(*args, **kwargs)
    return result, list(dict.fromkeys(str(w.message) for w in caught))


def _replay(source: str, messages: list[str]) -> None:
    for m in messages:
        warn(m if m.startswith(source) else f'{source}: {m}', stacklevel=4)


def _annotate_one(path: Path):
    return _recorded(annotate, path)


def _compare_one(args) -> tuple[list[dict], list[str]]:
    """The rows of one pair: one per superposition (``align_on``)."""
    model_file, reference, pdb, kwargs = args
    kwargs = {**kwargs, 'align_on': _modes(kwargs.get('align_on', 'peptide_centric'))}
    try:
        table, caught = _recorded(compare, model_file, reference, name=pdb, **kwargs)
        rows = [{**row, 'error': ''} for row in table.to_dict('records')]
    except Exception as e:
        rows = [{'PDB': pdb, 'error': f'{type(e).__name__}: {e}',
                 **_settings(**{**kwargs, 'align_on': mode})} for mode in kwargs['align_on']]
        caught = []
    for row in rows:
        row['warnings'] = ' | '.join(caught)
    return rows, caught


def _map(fn, items, n_jobs: int, progress: bool, desc: str) -> list:
    """``[fn(x) for x in items]``, in `n_jobs` processes (-1: one per CPU)."""
    items = list(items)
    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs

    def run(results):
        if progress:
            try:
                from tqdm.auto import tqdm
                results = tqdm(results, total=len(items), desc=desc)
            except ImportError:
                pass
        return list(results)

    if n_jobs == 1 or len(items) < 2:
        return run(map(fn, items))
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(items))) as pool:
        return run(pool.map(fn, items))


def compare_pairs(pairs: pd.DataFrame, n_jobs: int = 1, progress: bool = False,
                  grouped: bool = False, **kwargs) -> pd.DataFrame:
    """Score a table of pairs (``PDB``, ``model_file``, ``reference_file``; see :func:`pair_files`).

    Each reference is annotated once, however many models it has. Warnings from
    every pair are raised again here (as :class:`TCRStructWarning`, prefixed with the
    file name) and kept in the ``warnings`` column.

    Args:
        n_jobs: worker processes (``-1``: one per CPU).
        progress: show a progress bar (needs tqdm).
        grouped: columns as ``(group, column)`` (:func:`.compare.group_columns`).
        kwargs: passed to :func:`.compare.compare` (``min_identity``, ``align_on``,
            ``align_atoms``, ``angle_method``, ``dockq_method``, ``tm_method``;
            ``**V1`` for the ``'v1'`` metrics).
            ``align_on=['peptide_centric', 'pmhc']`` scores each pair in both
            superpositions, the metrics that need none once.

    Returns:
        One row per pair (per pair and superposition, for a list of ``align_on``):
        ``PDB``, ``model``, ``method``, ``model_file``,
        ``reference_file``, the metric columns (numbers), the superposition
        (:data:`.compare.SETTINGS`), ``error`` (empty when the pair was scored; a
        failed pair keeps its row with NaN metrics) and ``warnings``.
        ``result.attrs['versions']`` has the package versions.
    """
    if pairs.empty:
        out = pd.DataFrame(columns=['PDB', 'model', 'method', 'model_file',
                                    'reference_file', *COLUMNS[1:], *NEW_COLUMNS, *SETTINGS,
                                    'error', 'warnings'])
        out.attrs['versions'] = versions()
        return group_columns(out) if grouped else out
    ref_files = list(dict.fromkeys(pairs['reference_file']))
    annotated = {}
    for path, (frame, caught) in zip(ref_files, _map(_annotate_one, ref_files, n_jobs,
                                                     progress, 'references')):
        _replay(Path(path).name, caught)
        annotated[path] = frame
    tasks = [(m, annotated[r], k, kwargs) for k, m, r in
             pairs[['PDB', 'model_file', 'reference_file']].itertuples(index=False)]
    results = _map(_compare_one, tasks, n_jobs, progress, 'models')
    for (model_file, *_), (_, caught) in zip(tasks, results):
        _replay(Path(model_file).name, caught)

    out = _numeric(pd.DataFrame([row for rows, _ in results for row in rows]).reindex(
        columns=[*COLUMNS, *NEW_COLUMNS, *SETTINGS, 'error', 'warnings']))
    repeat = [len(rows) for rows, _ in results]
    model_files = pairs['model_file'].repeat(repeat).astype(str).to_numpy()
    out.insert(1, 'model', [_stem(Path(m)) for m in model_files])
    out.insert(2, 'method', [Path(m).parent.name for m in model_files])
    out.insert(3, 'model_file', model_files)
    out.insert(4, 'reference_file', pairs['reference_file'].repeat(repeat).astype(str).to_numpy())
    out.attrs['versions'] = versions()
    return group_columns(out) if grouped else out


def _numeric(table: pd.DataFrame) -> pd.DataFrame:
    """Metric columns as numbers (``peptide_reversed`` as nullable booleans)."""
    table = table.copy()
    for column in [*COLUMNS[1:], *NEW_COLUMNS]:
        if column not in table:
            continue
        if column == 'peptide_reversed':
            table[column] = table[column].astype('boolean')
        else:
            table[column] = pd.to_numeric(table[column], errors='coerce')
    return table


def compare_models(models, reference, grouped: bool = False, **kwargs) -> pd.DataFrame:
    """Score several models of one reference: one row per model, metrics as numbers.

        >>> table = compare_models({'af3': 'af3.cif', 'boltz': 'boltz.pdb'}, 'rcsb.pdb')
        >>> compare_models(models, 'rcsb.pdb', grouped=True)['docking']

    Args:
        models: ``{name: model}`` or a list of models (files or frames; a list is
            named by file name or frame name).
        reference: the reference structure (file or frame), annotated once.
        grouped: columns as ``(group, column)`` (:func:`.compare.group_columns`).
        kwargs: passed to :func:`.compare.compare`.

    Returns:
        A DataFrame indexed by model name (by model and ``align_on`` for a list of
        superpositions). ``result.attrs['versions']`` has the
        package versions. Unlike :func:`compare_pairs`, a failing model raises.
    """
    from .tcr import TCRFrame
    if not isinstance(models, dict):
        models = {(Path(m).stem if isinstance(m, (str, Path)) else m.name): m for m in models}
    reference = TCRFrame.ensure(reference)
    rows = {name: compare(model, reference, **kwargs) for name, model in models.items()}
    if isinstance(kwargs.get('align_on', 'peptide_centric'), str):
        table = _numeric(pd.DataFrame(rows).T)
        table.index.name = 'model'
    else:
        table = _numeric(pd.concat(rows, names=['model', None]))
    table.attrs['versions'] = versions()
    return group_columns(table) if grouped else table


def compare_dirs(model_dir: str | Path, reference_dir: str | Path, n_jobs: int = 1,
                 key: Callable[[Path], str] = pdb_key, exclude: tuple[str, ...] = (),
                 progress: bool = False, existing: str | Path | pd.DataFrame | None = None,
                 **kwargs) -> pd.DataFrame:
    """Score every model under `model_dir` against its reference in `reference_dir`.

    Models are paired by `key` (default :func:`pdb_key`, so ``7N2N_model_*`` all
    go to the 7N2N reference). Models without a reference are skipped with a
    warning and listed in ``result.attrs['unmatched']``. See :func:`compare_pairs`
    for the other arguments and the result.

    Args:
        existing: an earlier result (a CSV file or table) to carry on from: pairs
            already in it (same model and reference file) are not scored again, and
            the result is its rows followed by the new ones. A file that does not
            exist yet counts as empty.
    """
    models = find_structures(model_dir, exclude=exclude)
    refs = find_structures(reference_dir)
    pairs, unmatched = pair_files(models, refs, key=key)
    if unmatched:
        shown = ', '.join(p.name for p in unmatched[:5]) + (' ...' if len(unmatched) > 5 else '')
        warn(f'{len(unmatched)} models have no reference in {reference_dir}: {shown}')
    done = _existing(existing)
    if len(done) and kwargs.get('grouped'):
        raise ValueError('existing= needs the flat table; drop grouped=True and group after')
    if len(done):
        scored = set(zip(_resolved(done['model_file']), _resolved(done['reference_file'])))
        todo = [(m, r) not in scored for m, r in
                zip(_resolved(pairs['model_file']), _resolved(pairs['reference_file']))]
        pairs = pairs[todo]
    table = compare_pairs(pairs, n_jobs=n_jobs, progress=progress, **kwargs)
    if len(done):
        attrs = table.attrs
        table = pd.concat([done, table], ignore_index=True) if len(table) else done
        table.attrs = {**done.attrs, **attrs}
    table.attrs['unmatched'] = [str(p) for p in unmatched]
    table.attrs['skipped'] = len(done)
    return table


def _existing(existing) -> pd.DataFrame:
    """The rows of an earlier result, or an empty table."""
    if existing is None:
        return pd.DataFrame()
    if isinstance(existing, pd.DataFrame):
        return existing
    path = Path(existing)
    if not path.exists():
        return pd.DataFrame()
    table = _numeric(pd.read_csv(path))
    for column in ('error', 'warnings'):
        if column in table:
            table[column] = table[column].fillna('')
    missing = {'model_file', 'reference_file'} - set(table.columns)
    if missing:
        raise ValueError(f'{path} has no {sorted(missing)} columns; is it a compare_dirs result?')
    return table


def _resolved(paths) -> list[str]:
    return [str(Path(p).resolve()) for p in paths]


def key_pattern(pattern: str) -> Callable[[Path], str]:
    """A `key` function from a regex: its first group (or whole match) on the file name, upper-cased.

    ``key_pattern(r'^([0-9A-Za-z]{4})')`` is the default rule; ``key_pattern(r'_(\\w{4})_')``
    reads the ID from the middle of the name.
    """
    rx = re.compile(pattern)

    def key(path: Path) -> str:
        m = rx.search(Path(path).name)
        if not m:
            raise ValueError(f'{Path(path).name} does not match {pattern!r}')
        return (m.group(1) if rx.groups else m.group()).upper()
    return key


# ----------------------------------------------------------------- normalizing

def _normalize_one(args) -> tuple[str, list[str]]:
    src, dst, kwargs = args
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        frame, caught = _recorded(normalize, src, **kwargs)
        frame.write(dst)
        return '', caught
    except Exception as e:
        return f'{type(e).__name__}: {e}', []


def normalize_dir(input_dir: str | Path, output_dir: str | Path, fmt: str | None = None,
                  n_jobs: int = 1, progress: bool = False, **kwargs) -> pd.DataFrame:
    """Normalize every structure under `input_dir` into `output_dir`, same sub-directories.

    Args:
        fmt: output format, ``'pdb'`` or ``'cif'``; default keeps each file's own.
        kwargs: passed to :func:`.chains.normalize` (``keep_leader``, ``heavy``, ``complex``).

    Returns:
        A table of ``input``, ``output`` and ``error`` (empty on success).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    tasks = []
    for src in find_structures(input_dir):
        suffix = f'.{fmt}' if fmt else '.cif' if '.cif' in src.name.lower() else '.pdb'
        dst = output_dir / src.relative_to(input_dir).parent / (_stem(src) + suffix)
        tasks.append((src, dst, kwargs))
    results = _map(_normalize_one, tasks, n_jobs, progress, 'normalizing')
    for (src, *_), (_, caught) in zip(tasks, results):
        _replay(src.name, caught)
    errors = [e for e, _ in results]
    return pd.DataFrame({'input': [str(t[0]) for t in tasks],
                         'output': [str(t[1]) for t in tasks], 'error': errors})
