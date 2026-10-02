"""Train the Med-Nexus binary respiratory classifier from a CSV/JSON manifest.
Manifest columns: image_path,label,report[,patient_id,diagnosis].
"""
import argparse, os, json, random, numpy as np, torch
from torch.utils.data import DataLoader, random_split
from data.online_medical_dataset import OnlineMedicalDatasetManager, RealMedicalDataset
from models.multimodal_fusion import MedNexusDiagnosticModel

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--root-dir",default=None); ap.add_argument("--epochs",type=int,default=5); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--val-ratio",type=float,default=.15); ap.add_argument("--out",default="checkpoints/med_nexus.pt"); ap.add_argument("--device",default="auto"); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    seed_all(args.seed); device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    mgr=OnlineMedicalDatasetManager(); cases=mgr.load_manifest(args.manifest,args.root_dir); ds=RealMedicalDataset(cases)
    n_val=max(1,int(len(ds)*args.val_ratio)); n_train=len(ds)-n_val; train_ds,val_ds=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(args.seed))
    tr=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True); va=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False)
    model=MedNexusDiagnosticModel().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); loss_fn=torch.nn.CrossEntropyLoss()
    best=float("inf"); os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for b in tr:
            opt.zero_grad(); out=model(b["image"].to(device), [" ".join(e) for e in b["evidence"]], attention_mode="dense"); loss=loss_fn(out["logits"],b["label"].to(device)); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); val=0
        with torch.no_grad():
            for b in va:
                out=model(b["image"].to(device), [" ".join(e) for e in b["evidence"]], attention_mode="dense"); val+=loss_fn(out["logits"],b["label"].to(device)).item()
        val/=max(1,len(va)); print(f"epoch={epoch} train_loss={total/max(1,len(tr)):.4f} val_loss={val:.4f}")
        if val<best: best=val; torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"val_loss":val,"seed":args.seed},args.out)
    print(f"saved={args.out}")
if __name__=="__main__": main()
