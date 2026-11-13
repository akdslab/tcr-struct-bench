---
name: show-tcrpmhc
description: Show TCR-pMHC structures in 3D with tcrstruct - peptide core, CDRs and MHC helices marked and switchable, TCR-up / groove-from-above orientations, and predicted models overlaid on the reference colored by per-residue lDDT. Use when asked to view, visualize, overlay or compare TCR-pMHC structures in a notebook or as HTML.
---

# Show TCR-pMHC structures

```python
import tcrstruct as tcs
from tcrstruct import TCRFrame
pdb1 = TCRFrame.read('ref.pdb')
pdb1.view()                                               # core, six CDRs, MHC helices
pdb1.view(regions=['peptide_core', 'cdr3'], sequence=['C', 'D', 'E'])
pdb1.view(regions='all', hidden_regions=['peptide_centric', 'Va', 'Vb'])   # start some off
pdb1.view(vectors=True)                                   # docking-angle vectors + angles
tcs.view(pdb1, regions='cdr3', extra_regions={'site': pdb1.select(chains='C')})
v = tcs.view_comparison({'AF3': 'af3.cif', 'Boltz': 'boltz.pdb'}, pdb1)  # models on reference
v.save('view.html')                                       # any view -> standalone page
```

**Regions**
- `regions=` takes `True` (default set), names, groups (`cdrs`, `cdr3`, `tcr`, `pmhc`,
  `all`, ...) or `None`. Case does not matter.
- Click a region in the legend to switch it off or on; *CDRs* / *CDR alpha* / *CDR beta*
  switch several.

**Orientation**
- Views show the atoms as they are in the file. `tcr_up=True` opens turned TCR up, with
  the peptide N to C left to right; only the camera turns, never the coordinates.
- The *TCR up* and *Groove from above* buttons turn the camera; *Reset view* goes back.

**`view_comparison`**
- `superpose=True` is a Cα fit on all chains; `['A', 'C']` fits on the pMHC only, which
  shows docking differences; `False` keeps the files' frames. Predicted models usually
  sit about 90 Å away.
- This is the viewer's fit, not the benchmark's peptide-centric one.
- `values='lddt'` colors each model by its per-residue lDDT; `compact=True` gives a
  smaller page for many models.

Anything else goes to PDBFrame's `view` (`style`, `surface`, `colors`, ...); see
`tcrstruct docs pdbframe`. Every TCRFrame view option: `tcrstruct docs tcrframe` (Viewing).
