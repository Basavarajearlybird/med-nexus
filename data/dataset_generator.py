"""Unified dataset/loader entry point for Med-Nexus."""
import os, random, numpy as np, torch
from torch.utils.data import DataLoader
from data.online_medical_dataset import OnlineMedicalDatasetManager, RealMedicalDataset

class RespiratoryDatasetGenerator:
    def __init__(self, seed=42, use_online_dataset=True, manifest_path=None, root_dir=None, smoke=True):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.seed=seed; self.online_manager=OnlineMedicalDatasetManager()
        if manifest_path:
            self.online_manager.load_manifest(manifest_path, root_dir)
            self.is_smoke=False
        elif smoke and use_online_dataset:
            # Lazy-load smoke data; importing the package must never require network access.
            self.is_smoke=True
        else:
            self.is_smoke=False
            self.online_manager.downloaded_cases=[]
    def generate_synthetic_xray(self,label=1,image_size=224,perturbation="clean"):
        """Return a smoke-test PIL image. This is not synthetic clinical evidence; it is a cached smoke sample."""
        cases=self.online_manager.downloaded_cases
        if not cases:
            try:
                cases=self._ensure_smoke_cases()
            except Exception:
                from PIL import Image, ImageDraw
                img=Image.new("RGB",(image_size,image_size),128); d=ImageDraw.Draw(img); d.ellipse((35,35,image_size-35,image_size-35),outline=200,width=2)
                return img
        case=cases[label % len(cases)]
        ds=RealMedicalDataset([case],perturbation=perturbation)
        # Return unnormalized image for API/tests.
        img=case["image"].copy()
        if perturbation=="gaussian_blur":
            from PIL import ImageFilter; img=img.filter(ImageFilter.GaussianBlur(radius=3.5))
        elif perturbation=="crop_degrade":
            w,h=img.size; img=img.crop((int(.15*w),int(.15*h),int(.85*w),int(.85*h))).resize((w,h))
        return img.resize((image_size,image_size))

    def generate_patient_record(self,record_id=0,label=None,perturbation="clean",k_distractors=0,conflict=False):
        cases=self.online_manager.downloaded_cases
        if not cases:
            try: cases=self._ensure_smoke_cases()
            except Exception:
                from PIL import Image; return {"image":Image.new("RGB",(224,224),128),"label":0,"evidence":["Smoke-test placeholder only; not medical data."],"retrieved_evidence":["Smoke-test placeholder only; not medical data."],"conflict":conflict,"perturbation":perturbation}
        case=cases[record_id % len(cases)]
        ds=RealMedicalDataset([case],perturbation=perturbation,k_distractors=k_distractors,conflict=conflict)
        item=ds[0]
        return {"image":case["image"].copy(),"label":int(item["label"]),"evidence":item["evidence"],"retrieved_evidence":item["evidence"],"conflict":conflict,"perturbation":perturbation}

    def _ensure_smoke_cases(self):
        if self.online_manager.downloaded_cases: return self.online_manager.downloaded_cases
        try:
            return self.online_manager.download_smoke_samples()
        except Exception:
            from PIL import Image, ImageDraw
            cases=[]
            for i,label in enumerate([0,1,1,0]):
                img=Image.new("RGB",(224,224),120); d=ImageDraw.Draw(img);
                if label: d.ellipse((50,50,174,174),outline=210,width=5)
                else: d.rectangle((65,65,159,159),outline=170,width=3)
                cases.append({"id":i,"patient_id":f"procedural_{i}","label":label,"diagnosis":"smoke_only","image_path":"","image":img,"clinical_report":("Normal smoke-test image." if label==0 else "Abnormal smoke-test image.")})
            self.online_manager.downloaded_cases=cases
            self.is_smoke=True
            return cases

    def get_dataloader(self,batch_size=4,shuffle=True,perturbation="clean",k_distractors=0,conflict=False,num_samples=None):
        cases=self.online_manager.downloaded_cases
        if not cases and self.is_smoke: cases=self._ensure_smoke_cases()
        if not cases: raise RuntimeError("No dataset loaded. Supply manifest_path or enable smoke=True.")
        if num_samples is not None: cases=cases[:min(num_samples,len(cases))]
        return DataLoader(RealMedicalDataset(cases,perturbation=perturbation,k_distractors=k_distractors,conflict=conflict,seed=self.seed),batch_size=batch_size,shuffle=shuffle)
