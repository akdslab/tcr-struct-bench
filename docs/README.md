# Documentation

| page | what it covers |
|---|---|
| [tcrframe.md](tcrframe.md) | every attribute and method of a `TCRFrame`: chains, residues and their columns, regions, contacts, docking angles, views, checks, several complexes in a file |
| [metrics.md](metrics.md) | how a model is scored: matching, the superposition, every column, per-residue lDDT, the defaults and `'v1'` |
| [datasets.md](datasets.md) | building a set of structures: TCR3d / STCRDab / RCSB lists, downloads, checks, normalizing |
| [cli.md](cli.md) | every `tcrstruct` command |

TCRFrame is built on PDBFrame, whose own pages are
[pdbframe.md](https://github.com/akdslab/PDBFrame/blob/main/docs/pdbframe.md) (the
structure table, selections, writing, unobserved residues) and
[seqframe.md](https://github.com/akdslab/PDBFrame/blob/main/docs/seqframe.md)
(sequences and alignments). The sequence annotation (ANARCI, MHC genes and numbering)
comes from tcrkit: [api.md](https://github.com/akdslab/tcrkit/blob/main/docs/api.md).

These pages also come with the installed package: `tcrstruct docs` lists them with
PDBFrame's and tcrkit's, `tcrstruct docs metrics` prints one, and `tcs.docs('metrics')`
gives its path.
