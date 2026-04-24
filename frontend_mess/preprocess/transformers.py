"""
Pre-export monkey-patches for transformers >= 4.50.

Newer transformers builds 4D causal masks with `torch.vmap` and detects
packed-sequence layouts with `torch.diff`. Both break under torch.export
(torch 2.6): vmap hits NYI inside `is_contiguous(memory_format=...)`, and
`aten.diff` is not lowered by torch-mlir.

Call `apply_export_patches()` before importing a transformers model and
the causal-mask path becomes export-friendly without changing the mask
values for non-packed inputs.
"""

import transformers.masking_utils as _masking_utils


def _nonvmap_for_bhqkv(mask_function, bh_indices=True):
    def wrapped(batch_arange, head_arange, q_idx, kv_idx):
        if bh_indices:
            b = batch_arange.view(-1, 1, 1, 1)
            h = head_arange.view(1, -1, 1, 1)
            q = q_idx.view(1, 1, -1, 1)
            k = kv_idx.view(1, 1, 1, -1)
        else:
            b = batch_arange
            h = head_arange
            q = q_idx.view(-1, 1)
            k = kv_idx.view(1, -1)
        return mask_function(b, h, q, k)

    return wrapped


def apply_export_patches() -> None:
    _masking_utils._vmap_for_bhqkv = _nonvmap_for_bhqkv

    # Skip packed-sequence detection so `torch.diff` never enters the graph.
    # Safe for single-sequence inputs; packed layouts would need a different path.
    original_preprocess = _masking_utils._preprocess_mask_arguments

    def preprocess_without_packed(*args, **kwargs):
        # Runtime return is a 5-tuple; the upstream type annotation is stale.
        result = original_preprocess(*args, **kwargs)
        return (*result[:2], None, *result[3:])

    _masking_utils._preprocess_mask_arguments = preprocess_without_packed
