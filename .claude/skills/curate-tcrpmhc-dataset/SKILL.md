---
name: curate-tcrpmhc-dataset
description: Build a clean set of TCR-pMHC or pMHC structures with tcrstruct - list entries from TCR3d / STCRDab, filter by MHC class, release date and resolution, download from RCSB, verify each is a complete complex, split peptides fused to another chain (e.g. 6XCP), and normalize chain IDs and IMGT numbering. Use when collecting or cleaning structures for a benchmark or training set.
---

# Curate a TCR-pMHC dataset

```python
import tcrstruct as tcs
cx = tcs.sources.tcr3d_complexes('II')            # also tcr3d_pmhc(), stcrdab(), rcsb_info(ids)
sel = cx[(cx['Release_date'] >= '2022-01-01') & (cx['Resolution'] <= 3.0)]
sel.to_csv('set_2026-10-20.csv', index=False)     # freeze the list, with its date
tcs.sources.rcsb_download(sel['pdb_id'], 'raw/', n_jobs=8)   # skips files already there
checks = tcs.verify_dir('raw/', n_jobs=8)         # file, valid, failed, warned
```
CLI:
```bash
tcrstruct rcsb-download --tcr3d II -o raw/
tcrstruct verify raw/ -o checks.csv
tcrstruct normalize raw/ clean/
```

**Fix what fails, in this order**
1. **Peptide fused to another chain** (failed check names the chain): use
   `pf.split_fused_peptide()` (CLI `tcrstruct split-peptide in out`). This is
   `evidence == 'structure'`: the overhang really lies in the groove.
   `'sequence (possible)'` is only a hint; check it by eye.
2. **Class I without B2M** is a warning; `verify(..., require_b2m=True)` (CLI
   `--require-b2m`) makes it a failure.
3. **Several copies** (`copies` warning): downloads are biological assembly 1 by default
   (`assembly=None` for the deposited structure), and reading, `verify` and `normalize`
   keep complex 1 by contacts, naming the chains left out; an assembly can still hold
   two complexes (9RUP).
   Every copy: `tcs.find_complexes(path)` or CLI `tcrstruct complexes file.cif -o dir/`;
   one: `complex=n` (CLI `--complex n`).
4. **Contact failures** (peptide not in the groove, TCR not on the pMHC, chains not
   paired): drop the entry. It is not a proper complex.

**Then normalize:** `tcs.normalize_dir('raw/', 'clean/')` gives chains A-F and TCRs
numbered by IMGT; residues before the V domain are dropped unless `keep_leader=True`.

**Structure checks** (warnings): chain breaks (with the region, e.g. a CDR3 gap), the
MHC's groove disulfide (Cys11/Cys74 of G-ALPHA2 or G-BETA), the conserved TCR residues (Cys23, Trp41, Cys104, Phe/Trp118) and, in normalized files, the
TCR numbering against the recomputed IMGT.

**Quick checks:** `pf.verify()` (a TCRFrame) gives a table; `as_bool=True` gives True / False.
Use `kind='pmhc'` for structures without a TCR. Split a complex into TCR / pMHC / MHC / peptide
files with `pf.split(out_dir=...)` (CLI `tcrstruct split`).

The sources, their columns and the whole procedure: `tcrstruct docs datasets`; every command:
`tcrstruct docs cli`.
