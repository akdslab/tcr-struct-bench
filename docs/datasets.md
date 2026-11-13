# Building a data set

`tcs.sources` lists TCR-pMHC and pMHC structures from public databases and downloads
their files from the RCSB PDB. tcrstruct then checks each structure and normalizes it.
Nothing is downloaded until you ask, and nothing is cached: save the tables you use, so
that a set stays fixed.

```python
import tcrstruct as tcs

cx = tcs.sources.tcr3d_complexes('II')               # every class II TCR-pMHC in TCR3d
sel = cx[(cx['Release_date'] >= '2022-01-01') & (cx['Resolution'] <= 3.0)]
sel.to_csv('set_2026-11-13.csv', index=False)        # freeze the list, with its date
tcs.sources.rcsb_download(sel['pdb_id'], 'raw/', n_jobs=8)
checks = tcs.verify_dir('raw/', n_jobs=8)            # file, valid, failed, warned
tcs.normalize_dir('raw/', 'clean/', n_jobs=8)        # chains A-F, TCRs numbered by IMGT
```

The same from the command line:

```bash
tcrstruct rcsb-download --tcr3d II -o raw/     # also saves raw/tcr3d_complexes.csv
tcrstruct verify raw/ -o checks.csv -j 8
tcrstruct normalize raw/ clean/ -j 8
```

## Lists of structures

| function | source | one row per | main columns |
|---|---|---|---|
| `tcr3d_complexes(mhc_class=None)` | [TCR3d](https://tcr3d.ibbr.umd.edu), its `tcr_complexes_data.tsv` | TCR-pMHC entry | `pdb_id`, `mhc_class`, `TCR_name`, `Epitope`, `MHC_allele`, V/J genes, the six CDR sequences, TCR3d's docking angles, `Release_date`, `Resolution` |
| `tcr3d_pmhc(mhc_class=None)` | TCR3d, its *MHC class I / II chains* pages | pMHC entry, with or without a TCR | `pdb_id`, `mhc_class`, `mhc_allele_name`, `mhc_species`, `peptide`, `is_bound` (1: also has a TCR), `release_date`, `resolution`, `pmid` |
| `stcrdab(peptide_mhc_only=True)` | [STCRDab](https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred), its summary TSV | TCR in an entry | the chain IDs of each TCR, antigen and MHC (`Achain`, `Bchain`, `antigen_chain`, `mhc_chain1` / `mhc_chain2`), species, resolution, release date |
| `rcsb_info(pdb_ids)` | RCSB PDB | entry | `title`, `release_date`, `method`, `resolution`, `n_entities` |

`mhc_class` is `'I'`, `'II'` or None for both. `stcrdab(peptide_mhc_only=True)` keeps
alpha/beta TCRs bound to a peptide on an MHC; it drops CD1 and MR1, gamma/delta TCRs and
TCRs without an antigen.

## Downloading the files

`rcsb_download(pdb_ids, out_dir, fmt='cif', assembly=1, overwrite=False, n_jobs=4)`
downloads many entries in parallel and returns `pdb_id`, `file` and `status` per ID:

- `fmt`: `'cif'` (every entry has one), `'pdb'` (missing for large entries) or `'fasta'`
  (the deposited sequences).
- `assembly`: the biological assembly, 1 by default; an entry without it gets its
  deposited structure, with a warning. `None` gives the deposited structure (the
  asymmetric unit), which may hold several copies of the complex.
- Files keep RCSB's names (`1AO7-assembly1.cif`, `1AO7.cif`, `1AO7.pdb`). Files already
  there are skipped unless `overwrite=True`.
- A failed download is listed (`'failed: ...'`) and warned about, not raised.

Prefer mmCIF: it has each chain's deposited sequence, so residues without atoms keep
their place and the IMGT numbering is not shifted by a gap (see
[tcrframe.md](tcrframe.md), Residue columns). One entry straight into a TCRFrame:
`TCRFrame.from_rcsb('1AO7')`.

## Checking and fixing

`tcs.verify(path)` (or `pf.verify()`) checks that a structure is a complete TCR-pMHC
(`kind='pmhc'` for one without a TCR): one row per check, or True / False with
`as_bool=True`. `tcs.verify_dir(directory)` gives one row per file. Fix what fails in
this order:

1. **Peptide fused to another chain** (e.g. 6XCP): `pf.split_fused_peptide()` (CLI
   `tcrstruct split-peptide in.cif out.cif`) moves it into its own chain C.
2. **Class I without B2M** is a warning; `require_b2m=True` (CLI `--require-b2m`) makes
   it a failure.
3. **Several copies of the complex:** reading, `verify` and `normalize` keep complex 1 by
   contacts and name the chains left out. `tcs.find_complexes(path)` gives every copy
   (CLI `tcrstruct complexes file.cif -o dir/`), `complex=n` one of them.
4. **Contact failures** (peptide not in the groove, TCR not on the pMHC, chains not
   paired): drop the entry.

The structure checks are warnings: chain breaks (with their region, e.g. a gap in
CDR3), the conserved TCR residues (Cys23, Trp41, Cys104, Phe/Trp118) and, in normalized
files, the TCR numbering against a fresh IMGT numbering.

## Normalizing

`tcs.normalize(path)` / `tcs.normalize_dir(input_dir, output_dir)` rename the chains to
the standard IDs (A MHC heavy or class II alpha, B class II beta, C peptide, D TCR alpha,
E TCR beta, F B2M) and renumber the TCR chains by IMGT (111A is residue 111, insertion
code A). TCR residues before the V domain are dropped unless `keep_leader=True`. Write
`.cif` to keep the unobserved residues; a PDB file leaves them out, with a warning.

`pf.split(out_dir=...)` (CLI `tcrstruct split`) writes the TCR, pMHC, MHC and peptide,
or any region, as their own files.
