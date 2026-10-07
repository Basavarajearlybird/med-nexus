"""Evaluate a trained Med-Nexus checkpoint on a manifest."""
import argparse, json, torch
from data.online_medical_dataset import OnlineMedicalDatasetManager, RealMedicalDataset
from torch.utils.data import DataLoader
from models.multimodal_fusion import MedNexusDiagnosticModel
from evaluation.metrics import compute_comprehensive_metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--root-dir",default=None); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--device",default="auto"); ap.add_argument("--out",default="results/evaluation.json"); args=ap.parse_args()
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    mgr=OnlineMedicalDatasetManager(); ds=RealMedicalDataset(mgr.load_manifest(args.manifest,args.root_dir)); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    model=MedNexusDiagnosticModel().to(device); parameter_count=sum(p.numel() for p in model.parameters()); ck=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ck["model_state_dict"]); model.eval(); print(f"model_parameters={parameter_count:,}"); yt=[];yp=[];pr=[]
    with torch.no_grad():
        for b in dl:
            out=model(b["image"].to(device),[" ".join(e) for e in b["evidence"]],attention_mode="dense"); yt.extend(b["label"].tolist()); yp.extend(out["predictions"].tolist()); pr.extend(out["probabilities"].cpu().tolist())
    result=compute_comprehensive_metrics(yt,yp,pr); result["model_metadata"]={"parameter_count":parameter_count,"model_class":"MedNexusDiagnosticModel","vision_backbone":"DenseNet-121","text_backbone":"emilyalsentzer/Bio_ClinicalBERT","checkpoint":args.checkpoint}; print(json.dumps(result,indent=2))
    import os; os.makedirs("results",exist_ok=True); open(args.out,"w").write(json.dumps(result,indent=2))
if __name__=="__main__": main()
