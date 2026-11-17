# tcrstruct

`tcrstruct` (`import tcrstruct as tcs`) scores a predicted TCR-pMHC structure against its
experimental reference. Built on PDBFrame (structures) and tcrkit (sequence annotation).

Every metric has two versions: the `'v1'` metrics, all set by `**tcs.V1`
(`align_atoms='heavy'`; `angle_method`, `dockq_method`, `tm_method` = `'v1'`), and the
defaults: a Cα fit (`'ca'`), DockQ and TM-score `'v2'` and `'piercelab'` angles. Both are
pinned by tests to 1e-4 relative, angles to 0.01°; see `docs/metrics.md`. A
`CLAUDE.local.md`, if there is one, has private notes on top of this file.

## Scope: hard rules

- Only create, edit or delete files in this repo. Temporary files go in the scratchpad.
- PDBFrame (`../PDBFrame`) and tcrkit (`../tcrkit`) are read-only unless the user agrees.
  When asking, say why it can't be done here and why it belongs there.
  - PDBFrame: general protein features only.
  - tcrkit: sequence annotation (ANARCI, MHC genes, G-domain numbering, helices).
  - Here: anything on structures, and which chain is which (`tcs.detect_chains`, IDs A-F).
- The `'v1'` metric variants (no refit, CA-CA lDDT neighbours, `'v1'` TM-score
  and DockQ, identical-aa residue matching) are implemented here, never by changing
  PDBFrame defaults.

## Backward compatibility

Released (0.2.0 is the first public version): don't change existing functions,
arguments, defaults or outputs; deprecate with a `TCRStructWarning` first, and get
approval for any breaking change. Never change a metric value without a reason the tests
pin, and mark any breaking change **Breaking:** in its `CHANGELOG.md` entry.

## Versioning

The version lives only in `src/tcrstruct/__init__.py`. Each change bumps the patch
(`0.2.1`, ...) and gets a one- or two-line entry at the top of `CHANGELOG.md`.

## Commands

```bash
uv sync          # .venv with pdbframe and tcrkit from GitHub (pinned in pyproject.toml)
uv run pytest
```

## Tests

- Validation pairs in `tests/data/` (a few, never a whole data set), each with its
  expected CSV row and source in `tests/data/README.md`.
- `pytest.approx`: rel 1e-4 for scores, abs 0.01° for angles. Never loosen.
- Run the docs and notebooks' examples on public structures; no local paths in
  committed files.

## Warnings

Never silence warnings. Raise `TCRStructWarning` (`_warnings.warn`) naming the structure
and chain, and say what is missing and how to get round it.

## Style: match PDBFrame

`from __future__ import annotations`, single quotes, `__all__` in every module, private
helpers `_`-prefixed, short Google-style docstrings, functions return new objects. Keep
`docs/tcrframe.md` in sync with `TCRFrame`. No emoji.
