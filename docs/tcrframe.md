# TCRFrame reference

A `TCRFrame` is a [PDBFrame](../../PDBFrame/docs/pdbframe.md) of a TCR-pMHC (or pMHC)
complex. Everything PDBFrame does still works; this page lists what TCRFrame adds, then
the PDBFrame part.

```python
from tcrstruct import TCRFrame

s = TCRFrame.from_rcsb('6V0Y')
s                      # TCRFrame '6V0Y', MHC class I, chains A, C, D, E, F
```

## At a glance

| what | how |
|---|---|
| build | `TCRFrame.read(path)`, `TCRFrame.from_rcsb(id)`, `TCRFrame(pf)`; a multi-copy file gives complex 1, `complex=2` the next; all of them: `tcs.find_complexes(path)` |
| what it is | `s`, `print(s.report())`, `s.summary()`, `s.mhc_class`, `s.has_tcr` |
| chains | `s.chains`, `s.chain_table` (role, gene, allele, identity, J gene per chain) |
| sequences | `s.sequences`, `s.sequence('D')`, `s.sequence('peptide_core')`, `s.sequence('CDRB3')`, `s.peptide`, `s.cdrs` |
| residues, atoms | `s.residues` (with `imgt`, `region`), `s.df` |
| regions | `s.select(regions='cdr3')`, `s.regions['CDRB3']`, `s.masks['CDRB3']`, `tcs.list_regions()` |
| contacts | `s.contact_map()` (CDRs x peptide), `s.contact_map(grid=True)`, `tcs.plot_contact_map(s)`, `s.cdr_contacts()` (per CDR) |
| docking angles | `s.angles`, `s.docking_angles('v1')`, `s.docking_vectors()` |
| 3D | `s.view()`, `s.view(vectors=True)`, `tcs.view_comparison(models, s)` |
| score a model | `s.compare(reference)`; see [metrics.md](metrics.md) |
| check, fix, write | `s.verify()`, `s.split_fused_peptide()`, `s.normalize()`, `s.split(out_dir=)`, `s.write(path)` |

## What is done when, and how to read the tables

**On loading:**
- Chains are renamed to the standard IDs (table below).
- An `imgt` column gets each residue's IMGT label: TCR V domains by ANARCI, MHC groove
  domains by IMGT's G-DOMAIN numbering. For an mmCIF file both number each chain's
  deposited sequence, the unobserved residues (no atoms) included, so a gap does not
  shift the labels; the unobserved residues get labels too, in `s.residues`.
- An `original_chain` column keeps each chain's ID in the file.

**On first use:** regions, CDRs, angles and chain genes are computed the first time you ask
for them, then kept.

**Selections:** every PDBFrame method (`select`, `transform`, `copy`, `+`, ...) returns a
TCRFrame. A selection keeps the regions of the complex it came from.

| ID | chain |
|---|---|
| A | MHC heavy chain (class I) or MHC alpha (class II) |
| B | MHC beta (class II) |
| C | peptide |
| D | TCR alpha |
| E | TCR beta |
| F | B2M |

**IMGT labels** are strings:
- TCR V domain: `'1'` ... `'128'`, with insertions such as `'111A'`. After the V domain
  the labels continue `'129'`, `'130'`, ..., and before it they are `'-3'` ... `'-1'`.
- MHC: G-domain positions such as `'54A'`. Class I chain A has two G domains, each
  numbered from 1. A signal peptide, if the chain has one, is `'-k'` ... `'-1'`.
- Other chains (peptide, B2M): `'-'`.

**Residue columns** (`s.df` per atom, `s.residues` per residue):

| column | what it is | example (9RUP copy 2, TCR beta) |
|---|---|---|
| `residue_number`, `insertion_code` | the file's own numbering (author numbering): can start anywhere, skip numbers, carry insertion codes (`100A`) | `51`, then `52` ... `65` unobserved, then `66` |
| `observed_index` | the residue's 1-based position in its chain among the residues with atoms, in *this* frame: no gaps, recomputed after a selection; empty for unobserved residues | `51`, then empty, then `52` |
| `full_index` | its 1-based position in the chain's deposited sequence (mmCIF only, else empty): counts unobserved residues, does not change with a selection | `51`, `52` ... `65`, `66` |
| `observed` | whether it has atoms (`s.residues` only; every residue of `s.df` has) | `True`, `False` ..., `True` |
| `imgt` | its IMGT label (above), a string | `'64'`, `'65'` ..., `'80'` |
| `region` | the most specific region it is in (`s.residues` only) | `'CDRB2'`, `'CDRB2'`, `'Vb'` ..., `'Vb'` |

So `residue_number` is for talking about a residue as the file does, `observed_index` for
lining up the residues a frame has, `full_index` for where it sits in the full
sequence, and `imgt` for comparing positions across TCRs and MHCs. A plain PDBFrame
(`PDBFrame.read`) has all but `imgt` and `region`, which come with annotation.

## Building one

| | |
|---|---|
| `TCRFrame.read(path, name=None, complex=1)` | a PDB / mmCIF file; with several copies of the complex, only copy `complex` (see below) |
| `TCRFrame.from_rcsb(pdb_id, assembly=1, save=None, complex=1, metadata=False)` | download from the RCSB: one complex of biological assembly 1 (see below; `assembly=None, complex=None` for the whole deposited entry); `save=` also keeps the file; `metadata=True` also fetches the entry's title, release date, method and resolution into `s.metadata` |
| `TCRFrame(pf, complex=1)` | annotate a PDBFrame you already have (`pf` is not changed) |
| `TCRFrame.from_biopython(structure)`, `TCRFrame.from_sequences(seqs)` | from Bio.PDB, or sequences only |
| `TCRFrame.ensure(x)` | any of the above inputs, as a TCRFrame |

## Chains and sequences

| | gives |
|---|---|
| `s.chains` | chain IDs, `['A', 'C', 'D', 'E', 'F']` |
| `s.chain_table` | per chain: `role`, `gene`, `allele`, `identity`, `j_gene`, `j_identity`, `original_chain`, `length`, `sequence`. MHC and B2M: the gene, closest allele and identity to it; TCR: the V gene (`TRAV12-2`), its closest allele (`TRAV12-2*01`) and identity, then the J allele and its identity (ANARCI) |
| `s.mhc_class` | `'I'`, `'II'` or `None` |
| `s.has_tcr` | both TCR chains present |
| `s.sequences` | `{chain: sequence}` of the residues with atoms |
| `s.full_sequences` | `{chain: sequence}` with the unobserved residues: the deposited sequences of an mmCIF file (the same as `s.sequences` for PDB files). From PDBFrame |
| `s.unobserved` | the rows of `s.residues` without atoms (mmCIF files only), with their `imgt` and `region` |
| `s.sequence(name)` | the sequence of a chain (`'D'`) or of a region in one chain (`'peptide_core'`, `'cdrb3'`, `'Va'`); None if the structure lacks it |
| `s.peptide` | peptide sequence (`s.sequence('C')`) |
| `s.peptide_core_sequence` | `s.sequence('peptide_core')`: the whole peptide (class I), or the 9-mer closest to the MHC (class II) |
| `s.cdrs` | `{'CDRA1': 'DSAIYN', ..., 'CDRB3': ...}` |

## Residues and atoms

| | gives |
|---|---|
| `s.residues` | one row per residue: PDBFrame's columns (`observed`: has atoms; `full_index`) plus `imgt`, `original_chain` and `region` (the most specific region the residue is in). From mmCIF the unobserved residues are there too, in sequence order, labelled from the deposited sequence: their region is a CDR, V domain or the peptide by label. Selections keep them: by chain, residues or label regions directly, by masks and other regions when inside what is kept; `s.write('x.cif')` keeps them, PDB warns |
| `s.df` | the atom table, one row per atom, with `imgt` and `original_chain` |

```python
s.select(chains='D').residues.query("region == 'CDRA3'")   # CDR3 alpha with IMGT labels
```

## Regions

| region | what it is | used by the benchmark for |
|---|---|---|
| `peptide` | the whole peptide (C) | |
| `peptide_core` | the whole peptide (class I), or the 9-mer closest to the MHC (class II) | `Peptide`, the `'v1'` DockQ ligand |
| `peptide_centric` | residues of any chain within 10 Å of the peptide core | the superposition, `Peptide_Centric` RMSD |
| `CDRA1` `CDRA2` `CDRA3` | TCR alpha CDRs: IMGT 27-38, 56-65, 105-117 | CDR RMSDs, the `'v1'` DockQ receptor |
| `CDRB1` `CDRB2` `CDRB3` | the same on TCR beta | CDR RMSDs, the `'v1'` DockQ receptor |
| `helices` | MHC groove helices, at fixed IMGT positions (the Pierce lab's residues) | docking angles, orientation |
| `Va`, `Vb` | TCR V domains (IMGT 1-128) | |
| `TCRa`, `TCRb`, `MHC`, `B2M` | whole chains | |

| group | regions |
|---|---|
| `default` | peptide_core, six CDRs, helices (what `view()` marks) |
| `all` | peptide, peptide_core, peptide_centric, six CDRs, helices, Va, Vb |
| `cdrs`, `cdr1`, `cdr2`, `cdr3` | all CDRs, or one CDR on both chains |
| `cdras`, `cdrbs` | the three alpha / beta CDRs |
| `tcr`, `pmhc` | TCRa + TCRb; MHC + B2M + peptide |

Names ignore case, spaces, `_` and `-`, so `'cdra1'` is `CDRA1` and `'peptide-core'` is
`peptide_core`.

| | gives |
|---|---|
| `s.select(regions='cdr3')` | atoms of regions or groups, as a TCRFrame. Combines with `chains=`, `atoms=`, `residues=`, `mask=`, `invert=` |
| `s.regions['CDRB3']` | one region as a TCRFrame |
| `s.masks['CDRB3']` | one region as a boolean atom mask (one value per row of `s.df`) |
| `s.peptide_core` | the peptide core as a TCRFrame |
| `TCRFrame.describe()` | table of every region and group |
| `tcs.list_regions()` | every region: its chains, how it is defined (fixed IMGT positions, distance, whole chain) and its IMGT positions |
| `tcs.list_region_groups()` | every group and its regions |

```python
s.select(regions=['cdr3', 'peptide_core'], atoms='ca')
s.select(chains='E', regions='cdrs')
```

For the MHC's IMGT numbering residue by residue (G domain, strand or helix region,
topology), use `tcs.mhc.mhc_numbering(s)`; `tcs.mhc.mhc_match(s.sequences['A'])` gives
the chain's gene, closest allele and identity to it (tcrkit's `annotate_mhc`).

## Contacts

| | gives |
|---|---|
| `s.contact_map()` | every CDR residue within 5 Å of the peptide (closest heavy atoms): `row`, `column`, `distance`, and each side's chain, region and IMGT label |
| `s.contact_map(rows='cdrs', columns='helices', cutoff=4.0)` | any two regions or groups |
| `s.contact_map(grid=True)` | the same as a grid: contacting row residues x every column residue, NaN for no contact |
| `tcs.plot_contact_map(s)` | the grid as a heatmap, darker for closer |
| `s.contacts(other, cutoff=5.0)` | PDBFrame's residue pairs, unlabelled |

Labels: `'CDRB3 L110'` is region, residue and IMGT position. Peptide residues are residue and
position counted from the peptide core: `L1 L2 F3 ... V9` for the class I peptide
LLFGYPVYV; for class II, 1 is the core's first residue (P1), the flanks `-1, -2, ...`
before it and `10, 11, ...` after.

## Docking angles

| | gives |
|---|---|
| `s.angles` | `{'crossing_angle', 'incident_angle'}` in degrees, by the default method |
| `s.docking_angles(method='piercelab')` | the angles by a chosen method (below) |
| `s.docking_vectors(method='piercelab')` | the vectors behind them: `groove_axis`, `groove_normal`, `tcr_vector`, `rotation_axis`, centers, the helix mask |

| method | helices | TCR axis | range |
|---|---|---|---|
| `'piercelab'` (default; the Pierce lab's `tcr_docking_angle`) | fixed IMGT positions (`'imgt'`) | V domains only | signed, 0-180 |
| `'v1'` | Motif Atlas ranges (`'v1'`) | every shared IMGT label | folded, 0-90 |

Each part can be set on its own:
`s.docking_angles(helices='v1')`, `s.docking_angles('v1', signed=True)`.

## Viewing

| | |
|---|---|
| `s.view()` | 3D view with the peptide core, CDRs and MHC helices marked; legend toggles, sequence panel, *TCR up* / *Groove from above* buttons |
| `s.view(regions='all', hidden_regions=['peptide_centric'])` | which regions to mark, and which start switched off |
| `s.view(vectors=True)` | add the docking-angle vectors, labelled with the angles (`vectors='v1'` for that method) |
| `s.view(tcr_up=True)` | open turned TCR up, peptide left to right. Only the camera turns; by default the view shows the file's coordinates as they are |
| `s.tcr_up()` | a copy whose coordinates are turned that way (TCR up), e.g. to write to a file |
| `tcs.view_comparison(models, s)` | models superposed on `s`, colored by per-residue lDDT |

Other options (`sequence=`, `style=`, `surface=`, ...) go to PDBFrame's viewer.

## Scoring a model

`s.compare(reference)` (the same as `tcs.compare(s, reference)`) gives one row of benchmark
metrics: RMSDs, DockQ, lDDT, TM-score, docking angles and their errors. What each column
is, the superposition (`align_on=`, `align_atoms=`) and the `'v1'` settings are in
[metrics.md](metrics.md).

## Checking, fixing, writing

| | gives |
|---|---|
| `s.verify()` | one row per check: MHC class, required chains (a missing B2M is a warning; `require_b2m=True` makes it a failure), peptide, copies, extra chains, contacts, TCR docking (warns when few CDRs touch the pMHC, few CDR residues the peptide, or the TCR is reverse-docked), chain breaks, TCR anchor residues (Cys23, Trp41, Cys104, Phe/Trp118), TCR numbering. `kind='pmhc'` for no TCR; `complex=` which copy; `structure_checks=False` skips the last three |
| `s.cdr_contacts()` | per CDR: closest distance to the peptide and the MHC, and its residues touching each |
| `s.is_valid()` | True if nothing failed |
| `s.find_fused_peptide()` | a peptide fused to another chain (e.g. 6XCP: peptide on MHC beta) |
| `s.split_fused_peptide()` | that peptide moved to its own chain C |
| `s.normalize()` | TCRs renumbered by IMGT in `residue_number` / `insertion_code` (111A becomes 111 + `'A'`), ready to `write` |
| `s.split()` | `{'tcr', 'pmhc', 'mhc', 'peptide'}`: each part as a TCRFrame; `parts=['cdrs', 'peptide_core']` for regions; `out_dir='parts/'` writes `<name>_<part>.pdb` (`fmt='cif'`) |

## Several complexes in one file

An RCSB entry is the deposited structure (the asymmetric unit), which often holds two
or more copies of the complex. `TCRFrame.from_rcsb` takes two steps, both on by default:

- **`assembly=1`:** RCSB's biological assembly 1, what the depositors call the
  biological unit. Usually one complex, but not always (9RUP's assembly 1 is a dimer of
  two TCR-pMHCs). `assembly=None`: the deposited structure. An entry without an
  assembly file falls back to it, with a warning. `s.metadata['assembly']` records it.
- **`complex=1`:** of what was read, one complex by contacts. Reading a file with two
  chains of one kind keeps one complex, `complex=1` unless you pass another number, with a warning naming the
  chains left out. `s.metadata` records `complex`, `complex_chains` (the file's chain
  IDs) and `n_complexes`. `complex=None` keeps every chain, and the first of each kind
  by chain ID gets the standard ID, which can mix copies (2E7L does; `s.verify(complex=None)`
  then fails its `one complex` check).

`assembly=None, complex=None` gives the whole entry as deposited. `complex=` works the same
in `TCRFrame.read`, `from_rcsb`, `TCRFrame(pf)`, `tcs.verify`, `tcs.normalize` and the CLI
(`--complex N` or `--complex none`). `tcs.sources.rcsb_download` also takes assembly 1 by
default (CLI `--assembly N|none`).

| | gives |
|---|---|
| `tcs.complex_table(path)` | one row per complex: the file's chain of each part (`mhc`, `mhc_partner`, `peptide`, `tcr_alpha`, `tcr_beta`) |
| `tcs.find_complexes(path)` | each complex as a TCRFrame, named `<name>_1`, `<name>_2`, ... |
| `TCRFrame.read(path, complex=2)` | just one of them |
| `tcs.complexes.chain_contacts(pf)` | the chain x chain Cα contact counts the grouping uses |

Chains are grouped by contacts (Cα within 8 Å): each MHC groove chain takes the B2M or
class II beta it touches most, then the peptide, then the TCR alpha that touches that
pMHC most and the beta that touches that alpha most. Chains in no complex are named in a
warning. CLI: `tcrstruct complexes file.cif [-o dir/]`. Examples: `notebooks/05_verification.ipynb`.

## Summaries

| | gives |
|---|---|
| `s` | one line: name, MHC class, chains |
| `print(s.report())` | chains with roles and genes, peptide and core, CDRs (what `tcrstruct info` prints) |
| `s.summary()` | the main facts as a pandas Series |

## What comes from PDBFrame

These work as described in the [PDBFrame reference](../../PDBFrame/docs/pdbframe.md).
On a TCRFrame they return TCRFrames, with the standard chain IDs.

| | gives |
|---|---|
| `s.df` | the atom table |
| `s.name` | the label used in printouts and written files |
| `s.metadata` | facts about the whole structure, a dict (filled by `from_rcsb(..., metadata=True)`); kept by selections |
| `s.coords` | `(N, 3)` coordinates |
| `s.has_coordinates` | False for a sequences-only frame |
| `s.seqframe('D')` | one chain's sequence as a SeqFrame, ready to align |
| `s.select(chains=, residues=, atoms=, mask=, invert=)` | atoms matching every filter. `residues=(start, end)` uses the file's numbers; `atoms='ca'`, `'backbone'`, `'heavy'` or a list of names |
| `s.near(other, cutoff=5.0)` | whole residues of `s` within `cutoff` Å of `other`, e.g. `s.select(chains=['D', 'E']).near(s.peptide_core, 4.0)` |
| `s.contacts(other=None, cutoff=5.0)` | residue pairs in contact, with their closest distance; between chains if `other` is None |
| `s.rename_chains({'D': 'A'})` | chains renamed |
| `s.renumber(start=1)` | residues numbered 1, 2, ... per chain (insertion codes dropped) |
| `s.transform(R, t)` | coordinates moved: `xyz @ R.T + t` |
| `s + other` | two structures in one (chain IDs must differ) |
| `s.copy()` | a fully separate copy |
| `s.map_chains(other)` | `{my_chain: other_chain}`, found by sequence |
| `s.match(other)` | the atoms both share, in the same order: a pair of frames |
| `s.superpose(other, atoms='ca')` | `s` moved onto `other` by the best fit of their common atoms (PDBFrame's general fit; the benchmark's is `compare(align_on=)`) |
| `s.metrics.rmsd(ref)`, `.tm_score`, `.lddt`, `.dockq`, `.plddt`, `.summary` | PDBFrame's general scores. They are not the benchmark's variants; use `s.compare(ref)` for those, and `tcs.metrics` for its steps one by one (`match`, `superpose`, `rmsds`, `lddt`, ...) |
| `s.write('out.pdb')`, `s.to_string('cif')`, `s.to_fasta()`, `s.to_biopython()` | export |
| `PDBFrame.view(s, ...)`, `s.py3Dmol(...)` | PDBFrame's own viewer, without tcrstruct's regions and orientation buttons, or a py3Dmol view |
