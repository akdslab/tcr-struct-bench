---
name: inspect-tcrpmhc
description: Explain a TCR-pMHC or pMHC structure with tcrstruct - which chain is what (MHC gene/allele, class, B2M, peptide, TCR alpha/beta), the peptide core, CDR sequences, IMGT numbering of residues, docking angles, and region selections. Use when asked what is in a TCR-pMHC file, for CDRs or IMGT positions, or to select regions of such a structure.
---

# Inspect a TCR-pMHC structure

```python
import tcrstruct as tcs
from tcrstruct import TCRFrame
pdb1 = TCRFrame.read('7n2n.pdb')     # or TCRFrame.from_rcsb('7N2N'); a PDBFrame, annotated
pdb1                                 # name, MHC class, chains
print(pdb1.report())                 # species, chains (gene/allele), peptide, core, CDRs
pdb1.chain_table                     # role, gene, allele, species, identity, j_gene, ...
pdb1.species, pdb1.tcr_species, pdb1.mhc_species   # the complex's is the TCR's
pdb1.cdrs, pdb1.sequence('peptide_core'), pdb1.angles, pdb1.mhc_class
pdb1.contact_map()                   # CDR residues within 5 Å of the peptide (grid=True: a grid)
tcs.plot_contact_map(pdb1)           # ... as a heatmap; peptide positions count from the core
pdb1.select(regions='cdra3')         # a region as a TCRFrame; names ignore case
pdb1.residues                        # per residue: imgt label, original_chain, region
pdb1.select(chains='D').residues     # selections keep the regions of the whole complex
```
`TCRFrame(pf)` annotates a PDBFrame you already have; `pf` itself is unchanged. Every
PDBFrame method works and returns a TCRFrame. CLI: `tcrstruct info file.pdb`.

**Chains** are renamed to standard IDs, with the file's ID kept in `original_chain`:

| ID | chain |
|---|---|
| A | MHC heavy chain (class I) or alpha (class II) |
| B | MHC beta |
| C | peptide |
| D | TCR alpha |
| E | TCR beta |
| F | B2M |

**IMGT**
- The label is in the `imgt` column (`'111A'`), not `residue_number` (the file's
  numbering).
- Past the V domain the labels continue from 129; before it they are `'-k'..'-1'`;
  non-TCR chains get `'-'`.
- `tcs.normalize(pf)` writes IMGT into `residue_number` / `insertion_code`.
- ANARCI comes from tcrkit. CDR1 / CDR2 / CDR3 are IMGT 27-38 / 56-65 / 105-117; an
  insertion counts as its position (111A is 111).

**Regions**

| name | what it is |
|---|---|
| `peptide` | the whole peptide |
| `peptide_core` | the whole peptide (class I) or the 9-mer closest to the MHC (class II) |
| `peptide_centric` | residues within 10 Å of the peptide core (the superposition region) |
| `CDRA1`..`CDRB3` | the six CDRs |
| `helices` | the MHC groove helices |
| `Va`, `Vb` | the TCR V domains |
| `TCRa`, `TCRb`, `MHC`, `B2M` | whole chains |

Groups: `cdrs`, `cdr1`-`cdr3`, `cdras`, `cdrbs`, `tcr`, `pmhc`, `default`, `all`.
`TCRFrame.describe()` lists them all.

Every TCRFrame attribute and method: `tcrstruct docs tcrframe` (prints `docs/tcrframe.md`).
