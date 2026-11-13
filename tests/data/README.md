# Test data

Four TCR-pMHC complexes, each with its experimental structure and three predicted models:

| file | what |
|---|---|
| `{PDB}/rcsb.pdb` | the experimental structure, from the RCSB PDB entry |
| `{PDB}/{model}.pdb` / `.cif` | a predicted model of it |
| `{PDB}/{model}_processed.csv` | the prepared model (one row per atom, read with `tcs.read_processed`), which the expected `'v1'` values were scored from |
| `expected_metrics.csv` | the `'v1'` metrics of each pair, from an independent implementation |

Release dates are before (pre) or after (post) the models' training cutoff:

| PDB | cutoff | class |
|---|---|---|
| 1AO7 | pre-cutoff | I |
| 3QIB | pre-cutoff | II |
| 6ZKW | post-cutoff | I |
| 7T2D | post-cutoff | II |

Also `7N2N/rcsb.pdb` (from the RCSB PDB entry, no models): its
15-residue CDR3 alpha has the IMGT insertions 111A and 112A, for the normalization
tests and the notebook.

Also `2E7L/rcsb.cif`, downloaded from RCSB (https://files.rcsb.org/download/2E7L.cif), the
asymmetric unit: two copies of the 2C TCR on H-2Ld (no B2M). The first chain of each kind
by chain ID mixes the copies (MHC E, peptide P, TCR A and C), for the multi-copy tests:
by contacts the complexes are E, Q, A, D and F, P, B, C.

Also `9RUP/rcsb.cif`, downloaded from RCSB (https://files.rcsb.org/download/9RUP.cif):
a 4.1 Å cryo-EM dimer of two TCR-pMHCs (RCSB's assembly 1 holds both). Its TCRs barely
touch their pMHC (3 CDRs, 1 CDR residue on the peptide), for the `TCR docking` check.

Also `6XCP/rcsb.cif`, downloaded from RCSB (https://files.rcsb.org/download/6XCP.cif): its
peptide is fused to the MHC beta chain, for the fused-peptide and verification tests.

`expected_dockq_v2.csv`: DockQ v2 (https://github.com/wallnerlab/DockQ, MIT; run through
its `grouped_dockq`) on the 12 pairs, as written by `tcs.chains.scoring_frame`
(standard chain IDs, heavy atoms), with the TCR (D, E) and the pMHC (A, B, C, F) each
merged into one chain. `len1` / `len2` are the reference residues of the TCR and pMHC.

`expected_defaults.csv`: the default metrics of these pairs from an independent
implementation, in three superpositions (`full`, `pmhc`, `peptide_centric`), its chains
taken from the RCSB FASTAs (https://www.rcsb.org/fasta/entry/{PDB}). It leaves 3QIB out:
its chain assignment calls the 2B4 TCR beta chain an MHC class II beta chain.

The models for each are af3-tbm (`.cif`), boltz-2 and tcrmodel2 (`.pdb`). TCRmodel2
models are smaller (no B2M, about half the reference's atoms), so they test chains
that only partly match the reference.
