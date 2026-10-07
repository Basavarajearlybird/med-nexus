"""
Project 2: Retrieval Noise Experiment
Evaluates diagnostic robustness under varying levels of retrieved distractor passages.
"""

from typing import Dict, Any, List
import torch
from data.dataset_generator import RespiratoryDatasetGenerator
from services.inference_pipeline import MedNexusInferencePipeline
from evaluation.metrics import compute_comprehensive_metrics
import tqdm

def run_retrieval_noise_experiment(k_list: List[int] = [0, 1, 3, 5, 10], samples_per_k: int = 25, device: str = "auto", manifest_path=None) -> Dict[str, Any]:
    pipeline = MedNexusInferencePipeline(device=device)
    generator = RespiratoryDatasetGenerator(seed=42, manifest_path=manifest_path, smoke=manifest_path is None)
    
    results = {}
    
    for k in k_list:
        print(f"Running Retrieval Noise Benchmark for k={k}...")
        dataloader = generator.get_dataloader(batch_size=1, shuffle=False, perturbation="clean", k_distractors=k, conflict=False, num_samples=samples_per_k)
        
        y_true = []
        y_pred = []
        probs = []
        
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            img = batch["image"][0].unsqueeze(0).to(pipeline.device)
            label = batch["label"][0].item()
            evidence = batch["evidence"][0]
            
            with torch.no_grad():
                outputs = pipeline.model(image=img, text_list=evidence, attention_mode="dense", k_distractors=k)
                
            y_true.append(label)
            y_pred.append(outputs["predictions"].item())
            probs.append(outputs["probabilities"].squeeze(0).cpu().numpy().tolist())
            
        metrics = compute_comprehensive_metrics(y_true, y_pred, probs)
        results[f"k_{k}"] = metrics
        
    return {
        "experiment_name": "Retrieval Noise Stress Test",
        "results_by_k": results
    }
