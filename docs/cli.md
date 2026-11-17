# Command line

`tcrstruct <command> ...`; `tcrstruct <command> -h` lists every option. Warnings (a chain
not scored, a metric not computed, a model without a reference) go to stderr; `-q`,
before the command, hides them.

| command | what it does |
|---|---|
| `compare model reference` | scores one model; prints the row, or `-o row.csv` |
| `compare-dirs models/ references/ -o metrics.csv` | scores every model in a directory, one row each |
| `info structure` | chains, MHC class and genes, species, peptide core, CDRs, docking angles |
| `chains structure` | how each chain was detected, its species, and its new ID |
| `verify path [-o checks.csv]` | checks that a file (or each file in a directory) is a complete complex |
| `normalize input output` | standard chain IDs and IMGT-numbered TCRs, for a file or a directory |
| `split-peptide input output` | a peptide fused to another chain moved into chain C |
| `complexes structure [-o dir/]` | each copy of the complex in a multi-copy file |
| `split structure out_dir/` | the TCR, pMHC, MHC and peptide (or any region) as their own files |
| `rcsb-download [ids] -o dir/` | files from the RCSB PDB; `--tcr3d I / II / all` adds TCR3d's list |
| `docs [page]` | lists the documentation pages (PDBFrame's and tcrkit's too), or prints one (`--path`: its file) |
| `skills install` / `skills list` | the Claude Code skills, copied into `~/.claude/skills` (`--project`: `./.claude/skills`; `--deps`: PDBFrame's and tcrkit's too) |

## Scoring

```bash
tcrstruct compare model.cif reference.cif
tcrstruct compare-dirs predictions/ references/ -o metrics.csv -j 8
tcrstruct compare-dirs predictions/ references/ -o metrics.csv --align-on peptide_centric pmhc
tcrstruct compare-dirs predictions/ references/ -o metrics.csv --v1
```

`compare` and `compare-dirs` take the same scoring options; [metrics.md](metrics.md)
explains each:

| option | values (default first) |
|---|---|
| `--align-on` | `peptide_centric`, `full`, `pmhc`, `mhc`, `tcr`, `v_domains`, `cdrs`; several give one row per superposition |
| `--align-atoms` | `ca`, `heavy`: the atoms the superposition is fitted on |
| `--dockq` | `v2`, `v1` |
| `--tm-score` | `v2`, `v1` |
| `--angles` | `piercelab`, `v1`; one part only with `--angle-helices`, `--angle-axis`, `--angle-sign` |
| `--v1` | all of them `v1`, with `--align-atoms heavy` |
| `--min-identity` | 0.3: chains less identical to the reference are not scored |

`compare-dirs` searches both directories recursively and pairs files by PDB ID, the first
4 letters of the file name (`af3/7N2N_model_0.cif` goes with `7N2N.cif`); `--key-pattern
REGEX` takes it from the regex's first group instead. The `method` column is the
sub-directory; `--exclude rcsb` skips sub-directories by name; `-j` runs pairs in
parallel (`-1`: every CPU). Next to the CSV, `metrics.settings.json` records the options
and package versions.

## Structures

```bash
tcrstruct info 1ao7.cif
tcrstruct verify structures/ -o checks.csv -j 8        # --pmhc: no TCR expected
tcrstruct normalize raw/ clean/ --format cif -j 8      # --keep-leader, --heavy
tcrstruct complexes 2e7l.cif -o complexes/
tcrstruct split 1ao7.cif parts/ --parts tcr pmhc cdr3
tcrstruct split-peptide 6xcp.cif 6xcp_split.cif --drop-linker
```

`info`, `verify`, `normalize` and `split` keep complex 1 (by contacts) of a file with
several copies; `--complex N` takes another, `--complex none` keeps the whole file.

## Downloading

```bash
tcrstruct rcsb-download 1AO7 6XCP -o structures/
tcrstruct rcsb-download --tcr3d all -o structures/ -j 8    # every TCR-pMHC in TCR3d
tcrstruct rcsb-download 1AO7 -o structures/ --format pdb --assembly none
```

Files are biological assembly 1 in mmCIF by default; files already there are skipped.
See [datasets.md](datasets.md).

## Documentation and skills

```bash
tcrstruct docs                  # the pages of tcrstruct, PDBFrame and tcrkit, with their files
tcrstruct docs metrics          # print one
tcrstruct skills install        # into ~/.claude/skills, for Claude Code in every project
tcrstruct skills install --deps # ... and PDBFrame's and tcrkit's skills
tcrstruct skills install --project --force
```

The pages and skills come with the installed package, so they match its version.
