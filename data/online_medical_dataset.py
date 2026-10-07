"""
Med-Nexus medical dataset utilities.

The research path uses a user-supplied manifest for real data.  The tiny online
sample set is intentionally limited to smoke tests and is never presented as a
research dataset.
"""
from __future__ import annotations
import os, json, urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SMOKE_CASES = [
    {"url":"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BF2F-B7259A867D0C.jpeg","label":1,"diagnosis":"pneumonia","clinical_report":"Bilateral patchy opacities and ground-glass infiltrates throughout lower lung zones."},
    {"url":"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/0a72044e.jpeg","label":0,"diagnosis":"normal","clinical_report":"Clear lung fields without focal opacity, consolidation, pneumothorax, or pleural effusion."},
    {"url":"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/0C47C7D0-2E24-4066-A1EA-DD0A7F56A4F2.jpeg","label":1,"diagnosis":"infiltration","clinical_report":"Diffuse bilateral alveolar consolidation with prominent air bronchograms."},
    {"url":"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1.7._11.jpg","label":0,"diagnosis":"normal","clinical_report":"Lungs are fully expanded and clear. No focal consolidation or pleural effusion."},
]
DISTRACTORS = [
    "Degenerative changes of the thoracic spine without acute cardiopulmonary abnormality.",
    "No radiopaque foreign body identified in the shoulder soft tissues.",
    "Mild chronic aortic tortuosity without acute vascular abnormality.",
    "Calcified granuloma compatible with prior granulomatous exposure.",
    "Postoperative abdominal changes without acute thoracic finding.",
    "Mild cervical spondylosis without acute osseous abnormality.",
]

class OnlineMedicalDatasetManager:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_cases=[]

    def download_smoke_samples(self, max_samples=4):
        cases=[]
        for idx,item in enumerate(SMOKE_CASES[:max_samples]):
            path=self.cache_dir/f"smoke_{idx}.jpg"
            if not path.exists():
                try:
                    req=urllib.request.Request(item["url"], headers={"User-Agent":"Med-Nexus/1.0"})
                    with urllib.request.urlopen(req, timeout=20) as r: path.write_bytes(r.read())
                except Exception as exc:
                    raise RuntimeError(f"Could not download smoke image {idx}: {exc}") from exc
            img=Image.open(path).convert("RGB")
            cases.append({"id":idx,"patient_id":f"smoke_{idx}","label":int(item["label"]),"diagnosis":item["diagnosis"],"image_path":str(path),"image":img,"clinical_report":item["clinical_report"]})
        self.downloaded_cases=cases
        return cases

    def load_manifest(self, manifest_path, root_dir=None):
        path=Path(manifest_path)
        if not path.exists(): raise FileNotFoundError(path)
        if path.suffix.lower()==".csv": df=pd.read_csv(path)
        elif path.suffix.lower()==".json": df=pd.DataFrame(json.loads(path.read_text()))
        elif path.suffix.lower()==".jsonl": df=pd.read_json(path, lines=True)
        else: raise ValueError("Manifest must be CSV, JSON or JSONL")
        required={"image_path","label"}
        missing=required-set(df.columns)
        if missing: raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        root=Path(root_dir) if root_dir else path.parent
        cases=[]
        for i,row in df.iterrows():
            p=Path(str(row.image_path)); p=p if p.is_absolute() else root/p
            if not p.exists(): raise FileNotFoundError(f"Missing image: {p}")
            cases.append({"id":int(i),"patient_id":str(row.get("patient_id", i)),"label":int(row.label),"diagnosis":str(row.get("diagnosis","")),"image_path":str(p),"image":Image.open(p).convert("RGB"),"clinical_report":str(row.get("report", row.get("clinical_report","")))})
        self.downloaded_cases=cases
        return cases

class RealMedicalDataset(Dataset):
    def __init__(self,cases,transform=None,k_distractors=0,conflict=False,perturbation="clean",seed=42):
        self.cases=cases; self.k_distractors=k_distractors; self.conflict=conflict; self.perturbation=perturbation; self.seed=seed
        self.transform=transform or transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.cases)
    def _perturb(self,img):
        if self.perturbation=="gaussian_blur": return img.filter(ImageFilter.GaussianBlur(radius=3.5))
        if self.perturbation=="motion_blur":
            arr=np.asarray(img).astype(np.float32); k=9; kernel=np.zeros((k,k),np.float32); np.fill_diagonal(kernel,1.0/k)
            import cv2
            return Image.fromarray(np.clip(cv2.filter2D(arr,-1,kernel),0,255).astype(np.uint8))
        if self.perturbation=="crop_degrade":
            w,h=img.size; return img.crop((int(.15*w),int(.15*h),int(.85*w),int(.85*h))).resize((w,h),Image.Resampling.BILINEAR)
        return img
    def __getitem__(self,idx):
        c=self.cases[idx]; img=self._perturb(c["image"].copy()); label=int(c["label"]); report=c.get("clinical_report","")
        if self.conflict:
            others=[x for x in self.cases if int(x["label"])!=label and x.get("clinical_report")]
            if others: report=others[0]["clinical_report"]
        rng=np.random.default_rng(self.seed+idx)
        distractors=[]
        if self.k_distractors:
            distractors=list(rng.choice(DISTRACTORS,size=self.k_distractors,replace=self.k_distractors>len(DISTRACTORS)))
        return {"id":c["id"],"patient_id":c.get("patient_id",c["id"]),"image":self.transform(img),"label":torch.tensor(label,dtype=torch.long),"evidence":[report,*distractors],"report":report,"conflict":self.conflict,"perturbation":self.perturbation,"k_distractors":self.k_distractors}
