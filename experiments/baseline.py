from typing import Dict, Any
import time, numpy as np, torch
from data.dataset_generator import RespiratoryDatasetGenerator
from services.inference_pipeline import MedNexusInferencePipeline
from evaluation.metrics import compute_comprehensive_metrics

def run_baseline_experiment(num_samples=40, manifest_path=None, device="auto") -> Dict[str,Any]:
    pipeline=MedNexusInferencePipeline(device=device); gen=RespiratoryDatasetGenerator(seed=42, manifest_path=manifest_path, smoke=manifest_path is None)
    dl=gen.get_dataloader(batch_size=1,shuffle=False,num_samples=num_samples)
    y_true=[]; y_pred=[]; probs=[]; lat=[]
    for batch in dl:
        t=time.perf_counter()
        with torch.no_grad(): out=pipeline.model(batch["image"].to(pipeline.device), batch["evidence"][0], attention_mode="dense")
        lat.append((time.perf_counter()-t)*1000); y_true.append(int(batch["label"][0])); y_pred.append(int(out["predictions"][0])); probs.append(out["probabilities"][0].cpu().tolist())
    m=compute_comprehensive_metrics(y_true,y_pred,probs); m["avg_latency_ms"]=float(np.mean(lat)); return {"experiment_name":"Baseline Diagnostic Evaluation","num_samples":len(y_true),"metrics":m,"smoke_mode":gen.is_smoke}
