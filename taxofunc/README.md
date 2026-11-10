# TaxoFunc code

This directory contains the TaxoFunc-specific code for
`TaxoFunc-P2FM-ens-v2`.

## Contents

```text
taxofunc/
├── models/TaxoFunc-P2FM-ens-v2/
│   └── h_model.py          # PriorEncoderHP: 128 → 256 → 100
└── build_p2_block.py       # P2 block from the PICRUSt2 taxon→KO prior
```

`h_model.py` is the self-contained prior-encoder definition. `build_p2_block.py`
rebuilds the row-normalised PICRUSt2 table `P_ext` and the 128-dimensional P2
block, using a training-only PCA basis. It does not build tree or CLR blocks;
those are not part of the v2 model.

## Not included

The full ensemble-assembly script and the project's internal training and audit
scripts, result caches, and model weights are not included.

## v2 definition

```text
z* = 0.65 · h(P2) + 0.35 · z_FM
```

`h(P2)` is the average of five prior-encoder seeds; `z_FM` is the average of four
flow-matching sampling seeds. The fusion weight is fixed and is not learned
inside the network; it was selected on the validation split during model
development.
