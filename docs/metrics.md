# Metrics

`tcs.compare(model, reference)` scores one predicted TCR-pMHC against its experimental
structure and returns one row (a pandas Series; a DataFrame, one row per superposition, for
a list of `align_on`). This page says what is in that row, how
each number is computed, and which settings change it.

```python
import tcrstruct as tcs

row = tcs.compare('af3/1AO7.cif', 'rcsb/1AO7.pdb')                      # one model
table = tcs.compare_models({'af3': 'af3.cif', 'boltz': 'boltz.pdb'}, ref)   # one row per model
tcs.compare(model, ref, grouped=True)['docking']                             # one group of columns
tcs.compare(model, ref, align_on=['peptide_centric', 'pmhc'])                # two superpositions
table = tcs.compare_dirs('predictions/', 'references/', n_jobs=8)           # whole folders
tcs.per_residue_lddt(model, ref, by='region')                                # which parts are off
```

## How a pair is scored

```
model, reference ──> annotate (chains A-F, IMGT) ──> match residues (same amino acid only)
                                                            │
             ┌──────────────────────────────────────────────┤
             │                                              │
   one superposition (align_on, align_atoms)        no superposition needed
             │                                              │
   RMSDs: full, peptide, peptide-centric,           lDDT, docking angles,
   six CDRs, TCR / pMHC L-RMSD                      peptide direction,
   ('v1' DockQ and TM-score)                        DockQ v2, TM-score v2 (own fits)
```

1. **Annotate** both structures: standard chain IDs (A MHC heavy / class II alpha, B
   class II beta, C peptide, D TCR alpha, E TCR beta, F B2M) and IMGT labels.
2. **Match** residues chain by chain: a global alignment of the two sequences, keeping
   only aligned residues with the *same* amino acid, and the atoms both have. A chain
   less than 30 % identical to the reference (`min_identity=`) is not scored. A mutated
   residue is therefore left out of every RMSD.
3. **Superpose** the model once, on the region `align_on=`, fitted on its `align_atoms=`.
   There is no refit per region: every RMSD shows the error of that region *in this
   pose*.
4. **Score.** A metric that cannot be computed is NaN, with a `TCRStructWarning` that
   says why.

## The superposition

| `align_on=` (CLI `--align-on`) | fitted on | the RMSDs then show |
|---|---|---|
| `'peptide_centric'` (default) | residues of any chain within 10 Å of the reference's peptide core | the interface: the peptide, the CDRs and the MHC around them |
| `'full'` | every matched atom | the whole complex |
| `'pmhc'` | chains A, B, C, F | where the TCR sits on the pMHC (`TCR_LRMSD` is the pMHC-aligned L-RMSD) |
| `'mhc'` | chains A, B | as `'pmhc'`, and also the peptide's placement in the groove |
| `'tcr'` | chains D, E | where the pMHC sits relative to the TCR (`pMHC_LRMSD`) |
| `'v_domains'` | the TCR V domains (IMGT 1-128 of D and E) | as `'tcr'`, without the constant domains, which are often modelled poorly |
| `'cdrs'` | the six CDRs | the CDR loop shapes, not their placement |

`align_atoms=` (CLI `--align-atoms`): `'ca'` (default) fits the Cα atoms of that region;
`'heavy'` fits all its heavy atoms (`'v1'`). The RMSDs are always of
heavy atoms, except the two `*_LRMSD`, which are of Cα atoms, as docking L-RMSDs are.

**Several superpositions at once:** a list, `align_on=['peptide_centric', 'pmhc', 'cdrs']`
(CLI `--align-on peptide_centric pmhc cdrs`), gives one row per superposition. The columns
that do not depend on the pose (lDDT, angles, `peptide_reversed`, DockQ and TM-score
`'v2'`) are computed once and are the same in every row; `align_on` says which row is
which. `compare_models` then indexes by `(model, align_on)`, and `compare_pairs` /
`compare_dirs` have one row per pair and superposition.

## The columns

Columns are listed by group. `tcs.METRIC_GROUPS[group]` gives a group's columns, and
`grouped=True` (in `compare`, `compare_models`, `compare_pairs`, `compare_dirs`, or
`tcs.group_columns(table)`) returns `(group, column)` columns in this order. RMSD columns are
named by their region; the `'v1'` table calls two of them `Full_RMSD` and
`Peptide_RMSD` (`tcrstruct.compare.REFERENCE_NAMES`). **Pose** says whether
the value depends on the superposition (`align_on`, `align_atoms`).

### Global

| column | unit, better | pose | what it is |
|---|---|---|---|
| `Full` | Å, ↓ | yes | heavy-atom RMSD of every matched atom |
| `lddt` | 0-1, ↑ | no | lDDT of the matched heavy atoms: residue pairs with reference Cα atoms within 15 Å, distance kept within 0.5, 1, 2 and 4 Å |
| `tm_score` | 0-1, ↑ | `'v1'` only | TM-score of the matched Cα atoms, normalized by the matched residues. `'v2'` (default): on its own Kabsch fit of all of them; `'v1'`: in the superposition |

### Peptide

| column | unit, better | pose | what it is |
|---|---|---|---|
| `Peptide` | Å, ↓ | yes | heavy-atom RMSD of the peptide core: the whole peptide for class I, its 9-mer closest to the MHC for class II |
| `Peptide_Centric` | Å, ↓ | yes | heavy-atom RMSD of the residues within 10 Å of the peptide core (all chains) |
| `peptide_reversed` | True / False | no | the model's peptide runs backwards in the groove (C to N where the reference runs N to C) |

**How `peptide_reversed` is found.** Every peptide binds an MHC groove in the same
direction: its N-terminus at the start (N-terminal end) of the alpha1 helix and its C-terminus
at the end. So each structure is checked on its own
(`tcrstruct.angles.peptide_direction`): the cosine between the peptide core's first-to-last
Cα vector and the alpha1 helix's N-to-C vector. It is near +1 in a correct structure and
near -1 for a peptide modelled backwards. `peptide_reversed` is True when the model's sign
differs from the reference's. The reference is only a safeguard for an unusual structure;
`peptide_direction(model)` on a model alone already shows it. It uses no superposition and no
docking angle.

### CDRs

| column | unit, better | pose | what it is |
|---|---|---|---|
| `CDRA1` ... `CDRB3` | Å, ↓ | yes | heavy-atom RMSD of each CDR (IMGT 27-38, 56-65, 105-117), picked in each structure by its own IMGT labels and paired by rank |
| `Mean_CDR` | Å, ↓ | yes | mean of the six (NaN if one is) |

### Docking

| column | unit, better | pose | what it is |
|---|---|---|---|
| `DockQ` | 0-1, ↑ | `'v1'` only | DockQ (0.23 acceptable, 0.49 medium, 0.80 high). `'v2'` (default): [DockQ v2](https://github.com/wallnerlab/DockQ), TCR (D, E) against pMHC (A, B, C, F); `'v1'`: peptide core against the MHC and CDRs around it |
| `iRMSD` | Å, ↓ | `'v1'` only | DockQ's interface backbone RMSD |
| `LRMSD` | Å, ↓ | `'v1'` only | DockQ's ligand RMSD: the receptor fitted, the ligand's backbone measured. In `'v2'` the receptor is the side with more reference residues (usually the TCR; the pMHC when the TCR has no constant domains) |
| `TCR_LRMSD` | Å, ↓ | yes | Cα RMSD of the TCR (D, E). With `align_on='pmhc'`: the pMHC-aligned L-RMSD, where the TCR landed on the pMHC |
| `pMHC_LRMSD` | Å, ↓ | yes | Cα RMSD of the pMHC (A, B, C, F). With `align_on='tcr'`: the TCR-aligned L-RMSD |
| `fnat` | 0-1, ↑ | `'v1'` only | share of the reference's interface contacts (5 Å) the model has |
| `F1` | 0-1, ↑ | `'v1'` only | F1 of the model's contacts against the reference's |

### Angles

| column | unit | pose | what it is |
|---|---|---|---|
| `crossing_angle` | ° | no | the model's crossing angle: TCR alpha-beta vector against the groove's long axis |
| `incident_angle` | ° | no | the model's incident angle: TCR rotation axis against the groove normal |
| `crossing_angle_error`, `incident_angle_error` | °, ↓ | no | absolute difference of each angle from the reference's |

`'piercelab'` (default) as the Pierce lab's
[`tcr_docking_angle`](https://github.com/piercelab/tcr_docking_angle): fixed IMGT helix
positions, V-domain axis, signed (0-180°, so a reverse-docked TCR is above 90°). `'v1'`: Motif
Atlas helices, an axis over every shared IMGT label, folded to 0-90°.

### Contacts and bookkeeping

| column | what it is |
|---|---|
| `nat_correct`, `nat_total` | reference interface contacts the model has, and all of them (`fnat` = the ratio) |
| `nonnat_count`, `model_total`, `fnonnat` | model contacts the reference lacks, all model contacts, and the ratio |
| `clashes` | model residue pairs closer than 2 Å |
| `len1`, `len2` | the two DockQ sides' residue counts: TCR and pMHC (`'v2'`), ligand and receptor (`'v1'`) |
| `alignment_identity` | the lowest sequence identity of a chain to the reference |
| `align_on`, `align_atoms` | the superposition the row was scored in (pass them back to `compare` to score the same way) |

The package versions are kept off the row, in `row.attrs['versions']` (and
`table.attrs['versions']`, or `tcs.versions()`). The CLI writes them, with every setting,
to `<output>.settings.json` next to the CSV.

## Per-residue lDDT

`compare` gives one lDDT for the whole model. To see which parts are off:

```python
tcs.per_residue_lddt(model, ref)                 # one row per residue: chain, imgt, region, lddt
tcs.per_residue_lddt(model, ref, by='region')    # mean per region, worst first
tcs.view_comparison(model, ref)                  # 3D, colored by it
```

## Two versions: defaults and `'v1'`

Four parts of the scoring have two settings: the defaults, and `'v1'`, an earlier
definition kept so that older numbers can be reproduced. `tcs.V1` (CLI `--v1`) sets all
four.

```python
tcs.compare(model, reference)                     # defaults
tcs.compare(model, reference, **tcs.V1)           # everything v1
tcs.compare(model, reference, dockq_method='v1')  # one part at a time
```

| part | argument (CLI) | default | v1 (in `tcs.V1`) |
|---|---|---|---|
| superposition fit | `align_atoms=` (`--align-atoms`) | `'ca'` | `'heavy'` |
| DockQ | `dockq_method=` (`--dockq`) | `'v2'`: DockQ v2, TCR against pMHC; pairs residues and superposes by itself | `'v1'`: peptide core against the MHC within 10 Å plus the CDRs, in the superposition, no refit |
| TM-score | `tm_method=` (`--tm-score`) | `'v2'`: all matched Cα atoms on their own fit | `'v1'`: in the superposition |
| docking angles | `angle_method=` (`--angles`) | `'piercelab'` | `'v1'` |

Angle parts can be mixed: `angle_method={'helices': 'v1'}`, or on a frame
`s.docking_angles(helices='v1')`, `s.docking_angles('v1', signed=True)`.

These do not change between versions: the RMSDs (heavy atoms, one superposition, no
refit), lDDT, and residue matching (identical amino acids only).

## Notes

- **`'v1'` values:** with `**tcs.V1` and `align_on='peptide_centric'` the rows of the
  test pairs match an independent implementation's (rel 1e-4, angles 0.01°;
  `tests/test_benchmark.py`), and the defaults match another's in three superpositions
  (`tests/test_defaults.py`).
- **DockQ:** `'v2'` matches the DockQ program to 1e-4 on the test pairs. Reference: Mirabello
  and Wallner, *Bioinformatics* 40(10), btae586, 2024,
  [doi:10.1093/bioinformatics/btae586](https://doi.org/10.1093/bioinformatics/btae586).
- **TM-score:** both versions use L = the matched residues, so a model missing B2M or the
  constant domains is scored on the residues it has. `metrics.tm_score(..., length=n)`
  with the reference's length counts the missing ones against it instead.
- **Missing residues:** an mmCIF file lists every chain's deposited sequence, the
  unobserved residues (no atoms) included; PDBFrame keeps them in `pf.residues`
  (`observed` False), never in `pf.df`, so no metric sees them. That full sequence is what ANARCI
  and the MHC numbering see, and each label then goes on its residue, so a gap does not
  shift the IMGT labels (9RUP's second copy, CDRB2). A PDB file has no such table in
  PDBFrame yet, so it is numbered on the residues it has, as in the reference
  benchmark: labels after a gap can shift. A CDR whose residue counts then differ between
  model and reference cannot be paired by rank: that CDR and `Mean_CDR` are NaN, with a
  warning; the other RMSDs are still computed.
- **Multi-copy files** are cut to one complex first (complex 1 by contacts; see
  `docs/tcrframe.md`).
