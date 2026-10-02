"""
Project 2: Cross-Modal Conflict Experiment
Evaluates the GIB Conflict Detection module's ability to identify direct contradictions between image and text.
"""

from typing import Dict, Any
import torch
from data.dataset_generator import RespiratoryDatasetGenerator
from services.inference_pipeline import MedNexusInferencePipeline
import tqdm

def run_cross_modal_conflict_experiment(samples: int = 30, device: str = "auto", manifest_path=None) -> Dict[str, Any]:
    pipeline = MedNexusInferencePipeline(device=device)
    generator = RespiratoryDatasetGenerator(seed=42, manifest_path=manifest_path, smoke=manifest_path is None)
    
    print("Running Cross-Modal Conflict Resolution Test...")
    dataloader = generator.get_dataloader(batch_size=1, shuffle=False, perturbation="clean", k_distractors=0, conflict=True, num_samples=samples)
    
    conflict_detected_count = 0
    
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        img = batch["image"][0].unsqueeze(0).to(pipeline.device)
        evidence = batch["evidence"][0]
        
        with torch.no_grad():
            outputs = pipeline.model(image=img, text_list=evidence, attention_mode="dense")
            
        if outputs["conflict_detected"].item():
            conflict_detected_count += 1
            
    detection_rate = conflict_detected_count / float(samples)
    
    return {
        "experiment_name": "Cross-Modal Conflict Benchmark",
        "num_samples": samples,
        "conflict_detection_rate": detection_rate
    }
