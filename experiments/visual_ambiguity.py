from typing import Dict, Any
import torch, numpy as np
from data.dataset_generator import RespiratoryDatasetGenerator
from services.inference_pipeline import MedNexusInferencePipeline
from evaluation.metrics import compute_comprehensive_metrics

def run_visual_ambiguity_experiment(samples_per_cond=25, device="auto", manifest_path=None)->Dict[str,Any]:
    pipeline=MedNexusInferencePipeline(device=device); gen=RespiratoryDatasetGenerator(seed=42,manifest_path=manifest_path,smoke=manifest_path is None); results={}
    for cond in ["clean","gaussian_blur","motion_blur","crop_degrade"]:
        dl=gen.get_dataloader(batch_size=1,shuffle=False,perturbation=cond,num_samples=samples_per_cond); yt=[];yp=[];pr=[];conf=[]
        for b in dl:
            with torch.no_grad(): out=pipeline.model(b["image"].to(pipeline.device),b["evidence"][0],attention_mode="dense")
            yt.append(int(b["label"][0])); yp.append(int(out["predictions"][0])); pp=out["probabilities"][0].cpu().tolist(); pr.append(pp); conf.append(max(pp))
        m=compute_comprehensive_metrics(yt,yp,pr); m["mean_confidence"]=float(np.mean(conf)); results[cond]=m
    return {"experiment_name":"Visual Ambiguity Stress Test","results_by_condition":results,"smoke_mode":gen.is_smoke}
