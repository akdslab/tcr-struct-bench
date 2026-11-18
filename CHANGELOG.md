# Changelog

## 0.2.2 - 2026-11-18

- `compare_dirs(..., existing='metrics.csv')` (CLI `compare-dirs --resume`) scores only
  the pairs not already in that file; `verify` checks the MHC groove disulfide (Cys11 /
  Cys74 of G-ALPHA2 or G-BETA); `s.sequence_view()` shows the annotated sequences first.
- Needs PDBFrame 0.2.1 and tcrkit 0.1.11: MHC alleles now come from tcrkit's MHC Motif
  Atlas search (the same on the test structures; the first use downloads it).

## 0.2.1 - 2026-11-17

- The species of the TCR and of the MHC (human, mouse and, for the MHC, the 20 other
  species of TCR3d's structures), in `detect_chains`, `chain_table` and `s.species`.
- `s.kind` (`'tcrpmhc'`, `'pmhc'`, ...); a pMHC's `view()` marks no CDRs and does not warn.
- The repository is now `akdslab/tcrstruct` (was `tcr-struct-bench`); needs tcrkit 0.1.9.

## 0.2.0 - 2026-11-13

First public release.

- `TCRFrame`: a PDBFrame with the chains of a TCR-pMHC found and renamed (A-F), IMGT
  numbering of the TCR V domains (ANARCI) and the MHC groove domains, regions (peptide
  core, CDRs, MHC helices), contacts, docking angles and 3D views.
- Scoring a model against its reference: RMSDs (full, peptide, peptide-centric, six
  CDRs), DockQ, L-RMSDs, lDDT, TM-score and docking-angle errors, in any of seven
  superpositions; defaults and `'v1'` versions (`tcs.V1`).
- Normalizing (standard chain IDs, IMGT numbering), several complexes in one file,
  splitting, fused peptides, structure checks (`verify`), unobserved residues from mmCIF.
- Lists from TCR3d and STCRDab and batch RCSB downloads (`tcs.sources`).
- The `tcrstruct` command, and the docs and four Claude Code skills in the package.
