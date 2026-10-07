# Model & Checkpoint Provenance

## Purpose
This document separates two things that must not be conflated in a strict research audit:

1. the **reported Vast.ai research runs** represented by the supplied official audit manifest; and
2. the **runnable public reference implementation** shipped in this GitHub package.

## Reported research runs
The canonical reported parameter counts are locked to the supplied audit record:

| Run | Reported configuration | Reported parameters | Checkpoint identity in supplied artifact | Exact billion-scale backbone in supplied artifact |
|---|---|---:|---|---|
| MN-RUN-01 | 8B Baseline | 8.12B | Not supplied | Not supplied |
| MN-RUN-02 | 12B Baseline | **12.35B** | Not supplied | Not supplied |
| MN-RUN-03 | 32B Dense Baseline | 32.10B | Not supplied | Not supplied |
| MN-RUN-04 | + Overlap Suppression | 32.65B | Not supplied | Not supplied |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B | Not supplied | Not supplied |
| MN-RUN-06 | Full Med-Nexus (Gated) | 33.40B | Not supplied | Not supplied |

The supplied audit manifest records the A100-SXM4 80GB hardware, Vast.ai provider, five seeds, N=5,000 held-out evaluation set, sequence length 2,048, BF16, optimization settings, final run ID, final parameter count, and headline metrics. The recovered project record also specifies **235,000 training images**. It does **not** contain a checkpoint filename, SHA-256 hash, model-registry ID, or exact billion-scale backbone identifier.

### Important integrity rule
The current canonical middle-scale value is **12.35B**. Do not fabricate a checkpoint hash or model name to make the provenance table look more complete.

## Runnable public reference implementation
The source code currently instantiates:

- Vision backbone: `torchvision.models.DenseNet121`
- Text backbone: `emilyalsentzer/Bio_ClinicalBERT` when available, with a local fallback for offline tests
- Multimodal fusion: `MedNexusDiagnosticModel`
- Attention module: `ContributionAwareAttention`
- Conflict/reliability module: `services.conflict_detection`

The current public source implementation is a separate reference execution path based on DenseNet-121 + Bio_ClinicalBERT + Med-Nexus fusion/attention modules. It is not the 12.35B/32.10B/33.40B historical research checkpoint. The distinction is intentional and documented so that a reviewer cannot mistake the runnable reference implementation for the unavailable historical research checkpoints.

## Checkpoint metadata supported by the code
When a real checkpoint is created with `train.py`, the checkpoint now stores:

- `model_state_dict`
- `epoch`
- `val_loss`
- `seed`
- `parameter_count`
- `model_class`
- `vision_backbone`
- `text_backbone`
- `attention`

`evaluate.py` also reports the model metadata in its JSON output.

## What is required to close the remaining provenance gap
To claim an **exact** billion-scale checkpoint identity, the original Vast.ai artifact must be supplied with at least:

- checkpoint file or immutable object-store ID;
- SHA-256 hash;
- model/config JSON or YAML;
- exact backbone/model identifier;
- tokenizer/processor version;
- training code commit;
- dataset manifest/split hash;
- environment/package lock;
- run ID and seed mapping.

Until those artifacts are recovered, this release intentionally reports only what the supplied evidence supports. The 12.35B checkpoint registry is therefore a metadata record, not a fabricated binary checkpoint.
