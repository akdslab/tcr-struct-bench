"""The documentation and the Claude Code skills that come with the package.

    >>> tcs.docs()                       # {name: path} of every page, PDBFrame's and tcrkit's too
    >>> tcs.docs('metrics')              # the path of docs/metrics.md
    >>> tcs.install_skills()             # copy the skills into ~/.claude/skills
    >>> tcs.install_skills(project=True)   # ... or into ./.claude/skills
    >>> tcs.install_skills(deps=True)    # ... with PDBFrame's and tcrkit's skills

The pages are the repository's ``docs/`` folder, and the skills its
``.claude/skills/``. Both are put in the wheel (``tcrstruct/docs``,
``tcrstruct/skills``), so they match the installed version; in a source checkout they
are read from the repository itself. On the command line: ``tcrstruct docs [name]`` and
``tcrstruct skills install``.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

__all__ = ['SKILLS', 'DEPENDENCIES', 'docs', 'skills', 'install_skills']

#: The skills for users of the package (``.claude/skills/<name>/SKILL.md``).
SKILLS = ['score-tcrpmhc-models', 'inspect-tcrpmhc', 'show-tcrpmhc', 'curate-tcrpmhc-dataset']

_HERE = Path(__file__).resolve().parent


def _folder(package_dir: Path, installed: str, checkout: str) -> Path | None:
    """`installed` inside an installed package, else `checkout` in its source checkout."""
    for path in (package_dir / installed, package_dir.parents[1] / checkout):
        if path.is_dir():
            return path
    return None


#: The packages tcrstruct is built on, whose pages and skills come with them.
DEPENDENCIES = ['pdbframe', 'tcrkit']


def _module(name: str):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _dependency_docs(name: str) -> Path | None:
    module = _module(name)
    if module is None:
        return None
    return _folder(Path(module.__file__).resolve().parent, 'docs', 'docs')


def docs(name: str | None = None):
    """The documentation pages: ``{name: path}``, or the path of one page.

    tcrstruct's pages come first, then PDBFrame's (``pdbframe``, ``seqframe``: the
    PDBFrame every TCRFrame is built on) and tcrkit's (``api``: the sequence
    annotation), when their installed versions include them.

    Args:
        name: a page, e.g. ``'metrics'``, ``'tcrframe'``, ``'datasets'``, ``'cli'``,
            ``'seqframe'`` (with or without ``.md``).

    Raises:
        FileNotFoundError: no such page (the message lists them), or no pages at all.
    """
    pages = {}
    for folder in (_folder(_HERE, 'docs', 'docs'), *map(_dependency_docs, DEPENDENCIES)):
        if folder is not None:
            for path in sorted(folder.glob('*.md')):
                if path.stem != 'README':
                    pages.setdefault(path.stem, path)
    if not pages:
        raise FileNotFoundError('no documentation found: it comes with the tcrstruct wheel '
                                '(tcrstruct/docs) or a source checkout (docs/)')
    if name is None:
        return pages
    key = name.removesuffix('.md')
    if key not in pages:
        raise FileNotFoundError(f'no page {name!r}; the pages are {", ".join(pages)}')
    return pages[key]


def skills() -> dict[str, Path]:
    """``{name: folder}`` of the skills in :data:`SKILLS` (each holds a ``SKILL.md``)."""
    root = _folder(_HERE, 'skills', '.claude/skills')
    if root is None:
        raise FileNotFoundError('no skills found: they come with the tcrstruct wheel '
                                '(tcrstruct/skills) or a source checkout (.claude/skills)')
    return {name: root / name for name in SKILLS if (root / name / 'SKILL.md').is_file()}


def install_skills(target: str | Path | None = None, project: bool = False,
                   force: bool = False, deps: bool = False) -> dict[str, str]:
    """Copy the skills where Claude Code finds them.

    Args:
        target: the skills folder to copy into. Default: ``~/.claude/skills`` (every
            project), or ``./.claude/skills`` with `project`.
        project: install into the current directory's ``.claude/skills``.
        force: replace a skill of the same name that differs (by default it is kept).
        deps: also install PDBFrame's and tcrkit's skills (:data:`DEPENDENCIES`).

    Returns:
        ``{skill: status}``: ``'installed'``, ``'up to date'``, ``'updated'`` or
        ``'kept: differs (force=True to replace)'``; with `deps`, a dependency whose
        installed version has no skills is listed by name, with what to do.
    """
    if target is None:
        target = (Path.cwd() if project else Path.home()) / '.claude' / 'skills'
    target = Path(target)
    status = {}
    for name, source in skills().items():
        dest = target / name
        if not dest.exists():
            shutil.copytree(source, dest)
            status[name] = 'installed'
        elif _same_tree(source, dest):
            status[name] = 'up to date'
        elif force:
            shutil.rmtree(dest)
            shutil.copytree(source, dest)
            status[name] = 'updated'
        else:
            status[name] = 'kept: differs (force=True to replace)'
    for dep in DEPENDENCIES if deps else []:
        module = _module(dep)
        if module is None or not hasattr(module, 'install_skills'):
            version = getattr(module, '__version__', 'not installed')
            status[dep] = f'no skills in {dep} {version}: update it'
        else:
            status.update(module.install_skills(target, force=force))
    return status


def _same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    return not mismatch and not errors and all(
        _same_tree(a / d, b / d) for d in cmp.common_dirs)
