# License status

**TaxoFunc-specific code: no license granted yet; reuse rights pending.**

No top-level `LICENSE` file is provided for the TaxoFunc-specific code in this
repository. Until a license is added, no permission is granted to copy, modify,
distribute, or otherwise reuse that code.

## Vendored third-party code

The source under `models/scdiffusion_x/upstream/` is vendored from the
**scDiffusion-X** project. Two license files are retained:

- `models/scdiffusion_x/upstream/LICENSE` — **MIT License**, Copyright (c) 2026
  Erpai Luo (top-level upstream repository).
- `models/scdiffusion_x/upstream/scdiffusionX/LICENSE` — **BSD 3-Clause
  License**, Copyright (c) 2024, Erpai Luo (`scdiffusionX` package).

The exact file-by-file coverage boundary is **pending confirmation** with the
upstream project. Based on the directory layout, the top-level `LICENSE` appears
to cover repository-level files (for example `script/`), and the package
`LICENSE` appears to cover `scdiffusionX/src`. This is not a legal conclusion;
see `THIRD_PARTY_NOTICES.md` for provenance.

## No implied license

Nothing in this repository's documentation, figures, results, or code should be
read as granting a license for the TaxoFunc-specific material. The third-party
MIT and BSD-3-Clause terms apply only to the vendored upstream files they cover;
the exact coverage boundary is pending confirmation.
