# Tests

```bash
uv run pytest
```

The fit, DockQ, TM-score and the angles each have a `'v1'` version (all four with
`**tcs.V1`) and a default (`'v2'`, `'piercelab'`; see `docs/metrics.md`). Both are tested
on the 12 model-reference pairs in `data/` (see `data/README.md`): class I and II,
af3-tbm, boltz-2 and tcrmodel2 models, and chains that only partly match.

- `test_benchmark.py`, `'v1'`: each pair must match `expected_metrics.csv` in every
  column, scored from the raw files (which also tests chain detection and ANARCI) and
  from the prepared predictions (`*_processed.csv`, read with `tcs.read_processed`; the
  metrics alone).
- `test_metric_versions.py`, `'v2'`: DockQ must match DockQ v2 itself
  (`expected_dockq_v2.csv`) and TM-score pinned values; both must not depend on the
  model's pose or the superposition mode. The `'v2'` Cα fit is checked against pinned
  `full` / `pmhc` RMSDs. Also `**V1` against the defaults and the CLI flags.
- `test_defaults.py`, the defaults: every column of `expected_defaults.csv` (an
  independent implementation, 9 pairs, three superpositions), including `TCR_LRMSD`
  (the pMHC-aligned L-RMSD), and its angles against the `'v1'` angles.
- `test_scoring_tools.py`: `METRIC_GROUPS`, the `*_LRMSD` columns, `peptide_reversed`,
  `per_residue_lddt`, `compare_models`, unobserved residues and the CLI settings file.
- `test_complexes.py`: the two copies of 2E7L found by contacts, reading one by default
  (`complex=`), `split`, and the structure checks of `verify` (copies, one complex, chain
  breaks, MHC anchors, TCR anchors, TCR numbering).
- `test_tcr.py`: both angle methods, pinned, and `'piercelab'` against the Pierce lab's
  tool.

Tolerances are rel 1e-4 for scores and abs 0.01° for angles; `'v1'` angles carry float32
noise of about 1e-3° where the TCR α/β rotation is close to 180°. The other test modules
cover chains, MHC annotation, regions, views, batch scoring and verification.
