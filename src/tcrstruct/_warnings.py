"""The package's warning class, so its warnings can be shown, filtered or collected.

    >>> import warnings
    >>> warnings.simplefilter('ignore', tcs.TCRStructWarning)    # if you really want them gone
"""

from __future__ import annotations

import warnings

__all__ = ['TCRStructWarning', 'warn']


class TCRStructWarning(UserWarning):
    """Something about a structure the benchmark worked around: a chain not scored,
    a metric that could not be computed, a model without a reference, ..."""


def warn(message: str, stacklevel: int = 3) -> None:
    warnings.warn(message, TCRStructWarning, stacklevel=stacklevel)

