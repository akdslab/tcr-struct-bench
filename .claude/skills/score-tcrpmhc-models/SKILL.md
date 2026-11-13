---
name: score-tcrpmhc-models
description: Score predicted TCR-pMHC (or pMHC) models against experimental references with tcrstruct - RMSDs (full, peptide, peptide-centric, six CDRs), DockQ, lDDT, TM-score, docking-angle errors - for one pair or whole folders of models. Use when benchmarking AF3 / Boltz / TCRmodel2 / other predictions of TCR-pMHC complexes.
---

# Score TCR-pMHC models

```python
import tcrstruct as tcs
row = tcs.compare('model.cif', 'ref.pdb')                        # one pair -> Series
table = tcs.compare_models({'af3': 'af3.cif', 'boltz': 'b.pdb'}, 'ref.pdb')  # one ref, several models
table = tcs.compare_dirs('models/', 'refs/', n_jobs=8)           # folders -> DataFrame
table.groupby('method')[tcs.METRIC_GROUPS['docking']].mean()
tcs.compare('model.cif', 'ref.pdb', grouped=True)['cdrs']          # (group, column) columns
tcs.compare('model.cif', 'ref.pdb', align_on=['peptide_centric', 'pmhc'])  # one row per fit
tcs.per_residue_lddt('model.cif', 'ref.pdb', by='region')         # which regions are off
```
CLI: `tcrstruct compare-dirs models/ refs/ -o metrics.csv -j 8`.

**Pairing.** The PDB ID is the first 4 letters of the file name. So `af3/7N2N_model_0.cif`
and `boltz/7n2n_seed3.pdb` both go to `refs/7N2N.pdb`; `method` is the sub-folder and
`model` the file name. For other names, pass `key=tcs.batch.key_pattern(r'_(\w{4})_')`
(CLI `--key-pattern`).

**How it scores:**

| step | what happens |
|---|---|
| matching | chains by standard ID, residues by global alignment (identical aa only), heavy atoms |
| superposition | one peptide-centric fit (residues within 10 Å of the peptide core); every RMSD is in that pose, no refit |
| DockQ | default `dockq_method='v2'`: DockQ v2, TCR vs pMHC, own superposition; `'v1'`: peptide core vs (MHC within 10 Å + CDRs), not refitted |
| TM-score | default `tm_method='v2'`: all matched Cα on their own Kabsch fit; `'v1'`: current pose; L = matched residues in both |

Other fits: `align_on='full'` (all matched atoms), `'pmhc'` (chains A, B, C, F, so TCR
errors show up as TCR placement), `'mhc'` (A, B), `'tcr'` (D, E), `'v_domains'` (the TCR V domains,
IMGT 1-128, without the constant domains) or `'cdrs'` (the six CDRs: loop shape rather
than placement). A list of them scores every fit in one run (one row each, the pose-free
metrics computed once). The pMHC-aligned L-RMSD is `TCR_LRMSD` with
`align_on='pmhc'`; DockQ's own `LRMSD` and `iRMSD` do not depend on the fit (`'v2'`).
`peptide_reversed` flags a peptide modelled backwards in the groove (its direction
against the MHC alpha1 helix, in each structure; no reference angles needed).

**Reading the result**
- **Warnings:** check the `error` and `warnings` columns first. Warnings are raised as
  `tcs.TCRStructWarning`; never silence them.
- **Chains below `min_identity=0.3` are not scored** (warning). Different TCRs with
  identical constant domains still pass, so the wrong TCR can be scored silently.
- **No shared peptide** makes the peptide-centric fit impossible (error); use `align_on='full'`.
- **Missing residues:** mmCIF files are numbered on their deposited sequence
  (unobserved residues are in `pf.residues`, never in `pf.df`), so a gap does not shift the IMGT labels. PDB files are numbered on the
  residues they have; a CDR whose residue count then differs from the model's is NaN
  (with `Mean_CDR`), with a warning. Prefer mmCIF references.
- **Multi-copy references** are cut to complex 1 by contacts (warning).
- **Angle errors** are NaN when Cys 23/104 or the MHC helices are missing.
- **The fit, DockQ, TM-score and angles have two versions** (`tcrstruct docs metrics`).
  Defaults: the superposition fits Cα atoms (`align_atoms='ca'`), DockQ and TM-score `'v2'`,
  angles `'piercelab'` (as the Pierce lab's `tcr_docking_angle`: its helix residues,
  found by IMGT position, V-domain axis, 0-180°, so a reverse-docked model gets a large
  crossing error). For the `'v1'` metrics pass `**tcs.V1` (CLI `--v1`), or one
  part at a time: `align_atoms='heavy'`, `dockq_method='v1'`, `tm_method='v1'`,
  `angle_method='v1'` (Motif
  Atlas helices, whole-chain rotation axis, folded to 0-90°).
  Mix angle parts with a dict, e.g. `angle_method={'helices': 'v1'}` (CLI
  `--angle-helices`, `--angle-axis`, `--angle-sign`).

Column meanings: `tcrstruct docs metrics` (prints `docs/metrics.md`; `tcs.docs('metrics')`
is its path); walkthrough: `notebooks/01_scoring.ipynb` in the repository. The package versions are in
`table.attrs['versions']`; the CLI writes them with the settings to `<output>.settings.json`.
