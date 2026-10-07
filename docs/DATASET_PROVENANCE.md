# Dataset & Training-Corpus Provenance

## Reported training corpus

The recovered Med-Nexus project record specifies a training corpus of **235,000 medical images** for the historical billion-scale experiments.

This number is preserved as a **reported run-level project record**. The private/original dataset manifest, patient-level identifiers, storage paths, and raw image files are not included in this public release.

### Evaluation split

The supplied experiment record specifies a held-out paired image-text clinical evaluation set of **N=5,000**.

### Reproducibility boundary

The public repository provides the data-loader interface, manifest schema, experiment code, result records, and provenance documentation. It does not redistribute restricted clinical data or claim that the 235,000-image corpus can be reconstructed from this repository alone.

For a strict audit, the missing primary artifacts are:

1. original dataset manifest and split hash;
2. dataset storage/object identifier;
3. exact preprocessing configuration;
4. patient-level split record;
5. dataset checksum or immutable snapshot identifier.

The absence of these artifacts is explicitly recorded rather than replaced with fabricated identifiers.
