"""
Unit Tests for Med-Nexus Dataset Generator
"""

from data.dataset_generator import RespiratoryDatasetGenerator


def test_synthetic_xray_generation():
    generator = RespiratoryDatasetGenerator(seed=42)
    img_clean = generator.generate_synthetic_xray(label=1, perturbation="clean")
    assert img_clean.size == (224, 224)

    img_blur = generator.generate_synthetic_xray(label=1, perturbation="gaussian_blur")
    assert img_blur.size == (224, 224)


def test_patient_record_generation():
    generator = RespiratoryDatasetGenerator(seed=42)
    rec = generator.generate_patient_record(record_id=101, label=1, k_distractors=3, conflict=True)
    assert rec["label"] == 1
    assert rec["conflict"] is True
    assert len(rec["retrieved_evidence"]) == 4 # 1 true + 3 distractors
