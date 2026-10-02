"""
Med-Nexus FastAPI REST Microservice Application
Real-Time Stream-Processing Diagnostic API for Digital Respiratory Monitoring Platforms.
"""

import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import subprocess
import os

from api.schemas import DiagnosticRequest, DiagnosticResponse, HealthCheckResponse
from services.inference_pipeline import MedNexusInferencePipeline
from data.dataset_generator import RespiratoryDatasetGenerator

app = FastAPI(
    title="Med-Nexus Real-Time Diagnostic Microservice",
    description="Conflict-Resilient Multimodal Diagnostic Fusion & High-Efficiency Long-Context Sequence Tracking API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend / web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = MedNexusInferencePipeline(device="auto")
data_gen = RespiratoryDatasetGenerator(use_online_dataset=False, smoke=False)


@app.get("/", tags=["Root"])
def read_root():
    return {
        "title": "Med-Nexus Multimodal Diagnostic Service",
        "status": "Online",
        "target_journal": "Medical Image Analysis",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    return {
        "status": "Healthy",
        "system": "Med-Nexus Engine v1.0.0",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device": str(pipeline.device)
    }


@app.post("/analyze", response_model=DiagnosticResponse, tags=["Diagnostic Inference"])
def analyze_diagnostic_case(request: DiagnosticRequest):
    """
    Performs multimodal diagnostic inference over a synthetic or simulated case case.
    Accepts visual perturbation parameters and distractor configurations.
    """
    try:
        # Generate image based on request parameters
        img_pil = data_gen.generate_synthetic_xray(label=1, perturbation=request.visual_perturbation)
        result = pipeline.analyze(
            image=img_pil,
            query=request.query,
            patient_history=request.patient_history,
            attention_mode=request.attention_mode,
            k_distractors=request.k_distractors
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/upload", response_model=DiagnosticResponse, tags=["Diagnostic Inference"])
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    query: str = Form("Assess respiratory consolidation and pleural status"),
    patient_history: str = Form(None),
    attention_mode: str = Form("contribution_gated"),
    k_distractors: int = Form(0)
):
    """
    Performs diagnostic inference on an uploaded chest X-ray image file (JPEG/PNG).
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = pipeline.analyze(
            image=image,
            query=query,
            patient_history=patient_history,
            attention_mode=attention_mode,
            k_distractors=k_distractors
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {str(e)}")

@app.get("/retrieval/search", tags=["RAG Evidence"])
def retrieval_search(query: str, top_k: int = 3, k_distractors: int = 0):
    """
    Directly query the RAG medical evidence vector database.
    """
    try:
        results = pipeline.retrieval_service.retrieve_evidence(query, top_k=top_k, k_distractors=k_distractors)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments/run", tags=["Benchmarks"])
def run_experiments(background_tasks: BackgroundTasks):
    """
    Triggers the Med-Nexus benchmark suite to run in the background.
    """
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "run_all_experiments.py"))
    
    def run_script():
        subprocess.run(["python", script_path])
        
    background_tasks.add_task(run_script)
    
    return {"message": "Experiments launched in the background. Check results/ folder when completed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
