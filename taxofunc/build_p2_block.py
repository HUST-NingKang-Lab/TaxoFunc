"""Build the P2 side block used by TaxoFunc-P2FM-ens-v2.

This is a P2-only extraction of the project's original side-block builder and
PICRUSt2 feature library. Tree and CLR blocks are not part of the v2 model and
are intentionally not computed here.

P2 definition
-------------
    A_rel = counts / rowsum
    P     = A_rel @ P_ext              (P_ext row-normalised, 8,234 x 13,945)
    L     = log(P + 1e-6)
    L    -= L.mean(axis=1)             (CLR along the KO axis)
    P2    = ((L - TRAIN mean) @ TRAIN components) / TRAIN scale
    block = 0.1 * P2                   (128-d for the released model)

All bases are TRAIN-only. No KO truth is read.

Outputs (default taxofunc/cache/side_p2/):
    P_ext.npz            (8234, 13945) float32, row-normalised taxon->KO table
    side_blocks.npz      p2 (N, 128) float32
    sample_accession.npy (N,) same order as the A_tax columns
    meta.json            coverage, basis hashes, shapes
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

TAXOFUNC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TAXOFUNC)
SCX = os.path.join(ROOT, "models", "scdiffusion_x")
CACHE = os.path.join(SCX, "data", "mgm", "cache")

PICRUST_MODEL = os.path.join(
    ROOT, "baselines", "picrust2_full", "model", "picrust2_full_KO_prokaryotic")
P2_BASIS = os.path.join(CACHE, "p2_pca128.npz")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel():
    taxids = np.load(os.path.join(CACHE, "taxids.npy"))
    ko_names = np.load(os.path.join(CACHE, "ko_names.npy"))
    panel = np.array([str(t).split(".")[0] for t in taxids])
    return panel, np.array([str(k) for k in ko_names])


def build_p_ext(panel, ko_names, out_dir, chunk_rows=4000):
    """Rebuild the row-normalised PICRUSt2 taxon->KO table in panel/KO order."""
    fi_path = os.path.join(PICRUST_MODEL, "feature_index.txt")
    fn_path = os.path.join(PICRUST_MODEL, "function_index.txt")
    prof_path = os.path.join(PICRUST_MODEL, "feature_profiles.tsv.gz")

    features = [line.strip() for line in open(fi_path) if line.strip()]
    functions = [line.strip() for line in open(fn_path) if line.strip()]
    if len(set(features)) != len(features):
        raise RuntimeError("feature_index has duplicates")
    if len(set(functions)) != len(functions):
        raise RuntimeError("function_index has duplicates")

    fmap = {f: i for i, f in enumerate(features)}
    kmap = {k: i for i, k in enumerate(functions)}
    missing_ko = [k for k in ko_names if k not in kmap]
    if missing_ko:
        raise RuntimeError(
            f"{len(missing_ko)} panel KOs missing from PICRUSt2: {missing_ko[:5]}")
    ko_cols = [kmap[k] for k in ko_names]

    row_idx = np.array([fmap.get(t, -1) for t in panel], dtype=np.int64)
    matched = row_idx >= 0
    print(f"[p2] PICRUSt2 features {len(features)}, functions {len(functions)}; "
          f"panel matched {matched.sum()}/{len(panel)}", flush=True)

    P = np.zeros((len(panel), len(ko_names)), dtype=np.float32)
    reader = pd.read_csv(prof_path, sep="\t", index_col=0, chunksize=chunk_rows,
                         dtype={0: str}, low_memory=False)
    seen = 0
    for chunk in reader:
        idx = [fmap[str(f)] for f in chunk.index]
        local = {g: i for i, g in enumerate(idx)}
        sel_j, sel_i = [], []
        for j, pi in enumerate(row_idx):
            if pi >= 0 and pi in local:
                sel_j.append(j)
                sel_i.append(local[pi])
        if sel_j:
            arr = chunk.to_numpy(dtype=np.float32)[:, ko_cols]
            P[np.asarray(sel_j)] = arr[np.asarray(sel_i)]
        seen += len(chunk)
    print(f"[p2] read {seen} feature profile rows", flush=True)

    sums = P.sum(1, keepdims=True)
    P = np.divide(P, sums, out=np.zeros_like(P), where=sums > 0)
    np.savez(os.path.join(out_dir, "P_ext.npz"), P_ext=P)
    coverage = {
        "panel_taxids": int(len(panel)),
        "matched_taxids": int(matched.sum()),
        "unmatched_taxids": int((~matched).sum()),
        "ko_columns": int(len(ko_names)),
        "unmatched_rule": "zero_profile",
    }
    return P, coverage


_PREL = None


def external_table(p_ext_path):
    """Row-normalised external PICRUSt2 taxon->KO table, read once per process."""
    global _PREL
    if _PREL is None:
        p_ext = np.load(p_ext_path, allow_pickle=False)["P_ext"]
        _PREL = (p_ext / np.maximum(p_ext.sum(1, keepdims=True), 1e-12)).astype(np.float32)
    return _PREL


def p2_features(counts, basis_path, panel_taxids, p_ext_path):
    """Raw counts [N, T] -> standardised P2-PCA features [N, d] float32."""
    basis = np.load(basis_path, allow_pickle=True)
    if panel_taxids is not None:
        got = [str(t).split(".")[0] for t in panel_taxids]
        expected = [str(t) for t in basis["taxids"]]
        assert got == expected, (
            "[p2] panel/taxids mismatch: the basis was fitted on a different taxa "
            f"order ({len(got)} vs {len(expected)} columns)")
    X = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    X = np.asarray(X, dtype=np.float32)
    arel = X / np.maximum(X.sum(1, keepdims=True), 1.0)
    predicted_ko = arel @ external_table(p_ext_path)
    log_profile = np.log(predicted_ko + 1e-6)
    log_profile -= log_profile.mean(1, keepdims=True)
    return (((log_profile - basis["mean"]) @ basis["components"])
            / basis["scale"]).astype(np.float32)


def build(out_dir, chunk=2000, rebuild_p=False):
    os.makedirs(out_dir, exist_ok=True)
    panel, ko_names = load_panel()
    A = sp.load_npz(os.path.join(CACHE, "A_tax.npz")).tocsr()   # (8234, N)
    design = pd.read_csv(os.path.join(CACHE, "design.tsv"), sep="\t")
    samples = design["sample_accession"].astype(str).values
    if A.shape[1] != len(samples):
        raise RuntimeError(
            f"A_tax columns {A.shape[1]} != design rows {len(samples)}")
    n_samples = A.shape[1]
    print(f"[p2] A_tax {A.shape}, samples {n_samples}", flush=True)

    p_ext_path = os.path.join(out_dir, "P_ext.npz")
    if rebuild_p or not os.path.exists(p_ext_path):
        _, coverage = build_p_ext(panel, ko_names, out_dir)
    else:
        coverage = {"reused": True}
    print(f"[p2] P_ext ready: {p_ext_path}", flush=True)

    basis = np.load(P2_BASIS, allow_pickle=True)
    if [str(t) for t in basis["taxids"]] != list(panel):
        raise RuntimeError("P2 basis taxid order mismatch")
    d_p2 = int(basis["components"].shape[1])
    p2 = np.zeros((n_samples, d_p2), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        counts = A[:, start:stop].T.tocsr()          # (n, 8234)
        p2[start:stop] = p2_features(
            counts, P2_BASIS, panel, p_ext_path) * 0.1
        if (start // chunk) % 2 == 0:
            print(f"[p2] {stop}/{n_samples} ({time.time() - t0:.1f}s)", flush=True)

    np.savez_compressed(os.path.join(out_dir, "side_blocks.npz"), p2=p2)
    np.save(os.path.join(out_dir, "sample_accession.npy"), samples)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_samples": int(n_samples),
        "blocks": ["p2"],
        "dims": {"p2": d_p2},
        "scale": 0.1,
        "coverage": coverage,
        "sources": {
            "p2_basis": _sha(P2_BASIS),
            "picrust_features": _sha(os.path.join(PICRUST_MODEL, "feature_index.txt")),
            "picrust_functions": _sha(os.path.join(PICRUST_MODEL, "function_index.txt")),
            "picrust_profiles": _sha(os.path.join(PICRUST_MODEL, "feature_profiles.tsv.gz")),
            "A_tax": _sha(os.path.join(CACHE, "A_tax.npz")),
        },
        "leakage": {
            "fitted_on": "TRAIN only for the P2 basis",
            "picrust_functional_truth_used": False,
            "input": "taxonomy counts only",
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[p2] wrote {out_dir}; dims {meta['dims']}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(TAXOFUNC, "cache", "side_p2"))
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--rebuild-p", action="store_true")
    args = ap.parse_args()
    build(args.out_dir, chunk=args.chunk, rebuild_p=args.rebuild_p)


if __name__ == "__main__":
    main()
