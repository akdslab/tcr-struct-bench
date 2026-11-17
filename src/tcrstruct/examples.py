"""The example structures: the validation pairs in the repository's ``tests/data``.

    >>> tcs.example('1AO7', 'rcsb.pdb')          # a path
    >>> TCRFrame.read(tcs.example('1AO7', 'af3-tbm.cif'))
    >>> tcs.example()                            # the folder

They come with a source checkout (``uv sync`` installs the package from it), not with
a plain ``pip install``; ``tests/data/README.md`` lists them and their sources.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ['EXAMPLE_DATA', 'example']

#: The example folder: ``tests/data`` next to ``src/`` in a checkout.
EXAMPLE_DATA = Path(__file__).resolve().parents[2] / 'tests' / 'data'


def example(*parts: str) -> Path:
    """A file (or folder) of :data:`EXAMPLE_DATA`, e.g. ``example('1AO7', 'rcsb.pdb')``.

    Raises:
        FileNotFoundError: it does not exist (also: no checkout), with what there is.
    """
    path = EXAMPLE_DATA.joinpath(*parts)
    if path.exists():
        return path
    if not EXAMPLE_DATA.is_dir():
        raise FileNotFoundError(f'no example data at {EXAMPLE_DATA}: it comes with a source '
                                'checkout of tcrstruct (tests/data)')
    entries = sorted(p.name for p in EXAMPLE_DATA.iterdir() if p.is_dir())
    raise FileNotFoundError(f'{path} does not exist; the examples are {", ".join(entries)}')
