# Changelog

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
