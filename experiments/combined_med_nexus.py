"""
Combined Med-Nexus Multi-Failure Stress Test
Evaluates the full pipeline under all degradation modes simultaneously:
Retrieval Noise (k=5) + Visual Ambiguity (Gaussian Blur) + Cross-Modal Conflict + CRPA Attention.
"""

from typing import Dict, Any
import torch
from data.dataset_generator import RespiratoryDatasetGenerator
from services.inference_pipeline import MedNexusInferencePipeline
from evaluation.metrics import compute_comprehensive_metrics
import tqdm

def run_combined_med_nexus_experiment(samples: int = 30, device: str = "auto", manifest_path=None) -> Dict[str, Any]:
    pipeline = MedNexusInferencePipeline(device=device)
    generator = RespiratoryDatasetGenerator(seed=42, manifest_path=manifest_path, smoke=manifest_path is None)
    
    print("Running Combined Multi-Failure Stress Test...")
    dataloader = generator.get_dataloader(
        batch_size=1, 
        shuffle=False, 
        perturbation="gaussian_blur", 
        k_distractors=5, 
        conflict=True, 
        num_samples=samples
    )
    
    y_true = []
    y_pred = []
    probs = []
    conflict_detected_count = 0
    
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        img = batch["image"][0].unsqueeze(0).to(pipeline.device)
        label = batch["label"][0].item()
        evidence = batch["evidence"][0]
        
        with torch.no_grad():
            outputs = pipeline.model(image=img, text_list=evidence, attention_mode="contribution_gated", k_distractors=5)
            
        y_true.append(label)
        y_pred.append(outputs["predictions"].item())
        probs.append(outputs["probabilities"].squeeze(0).cpu().numpy().tolist())
        
        if outputs["conflict_detected"].item():
            conflict_detected_count += 1
            
    metrics = compute_comprehensive_metrics(y_true, y_pred, probs)
    detection_rate = conflict_detected_count / float(samples)
    
    return {
        "experiment_name": "Combined Multi-Failure Robustness Benchmark",
        "num_samples": samples,
        "med_nexus_accuracy": metrics["accuracy"],
        "conflict_detection_rate": detection_rate,
        "full_metrics": metrics
    }
