from typing import Dict,Any
import torch,time
from services.inference_pipeline import MedNexusInferencePipeline

def run_attention_efficiency_experiment(seq_length=16,num_patients=40,device="auto") -> Dict[str,Any]:
    pipeline=MedNexusInferencePipeline(device=device); dev=pipeline.device; x=torch.randn(1,seq_length,256,device=dev); modes=["dense","sliding_window","naive_suppression","contribution_gated"]; results={}
    for mode in modes:
        times=[]; stat=None
        for _ in range(num_patients):
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            t=time.perf_counter()
            with torch.no_grad(): _,stat=pipeline.model.crpa_attention(x,mode=mode)
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            times.append((time.perf_counter()-t)*1000)
        results[mode]={"avg_latency_ms":float(sum(times)/len(times)),"edge_retention_ratio":stat["retention_ratio"],"sparsity_ratio":stat["sparsity_ratio"],"suppressed_edges":stat["suppressed_edges"],"total_edges":stat["total_edges"],"device":str(dev),"note":"This benchmark measures the current dense score computation plus gating; edge suppression alone does not prove FLOP savings."}
    return {"experiment_name":"Attention Efficiency and Sparsity Benchmark","seq_length":seq_length,"num_samples":num_patients,"results_by_mode":results}
