"""
Med-Nexus Master Experiment Orchestrator
Runs complete diagnostic benchmark suite: Baseline, Retrieval Noise, Visual Ambiguity,
Cross-Modal Conflict, CRPA Attention Efficiency, and Combined Multi-Stress Test.
Generates publication plots and summary reports for Medical Image Analysis journal submission.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.baseline import run_baseline_experiment
from experiments.retrieval_noise import run_retrieval_noise_experiment
from experiments.visual_ambiguity import run_visual_ambiguity_experiment
from experiments.cross_modal_conflict import run_cross_modal_conflict_experiment
from experiments.attention_efficiency import run_attention_efficiency_experiment
from experiments.combined_med_nexus import run_combined_med_nexus_experiment
from evaluation.report_generator import ReportGenerator


def run_all(smoke_mode: bool = False):
    """Runs complete experiment suite (Experiments 1..6) and outputs benchmark plots & markdown report."""
    print("================================================================")
    print("      MED-NEXUS COMPREHENSIVE EXPERIMENTAL BENCHMARK SUITE       ")
    print("      Target Journal: Medical Image Analysis (ScienceDirect)    ")
    if smoke_mode:
        print("      [SMOKE MODE: Fast Validation Active]                      ")
    print("================================================================")
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    # Use the configured report generator when available; keep experiment orchestration deterministic.
    # For now we'll just run the experiment functions.
    
    # Scale down sizes for smoke mode
    num_samples = 4 if smoke_mode else 40
    samples_per_k = 2 if smoke_mode else 25
    samples_per_cond = 2 if smoke_mode else 25
    conflict_samples = 4 if smoke_mode else 30
    combined_samples = 4 if smoke_mode else 30

    # 1. Baseline Experiment
    print("\n[1/6] Running Baseline Diagnostic Evaluation...")
    baseline_res = run_baseline_experiment(num_samples=num_samples)
    print(f" -> Baseline Accuracy: {baseline_res['metrics']['accuracy']*100:.2f}%, ECE: {baseline_res['metrics']['ece']:.4f}")

    # 2. Project 2 Retrieval Noise Experiment
    print("\n[2/6] Running Project 2 Retrieval Noise Stress Test...")
    k_vals = [0, 1] if smoke_mode else [0, 1, 3, 5, 10]
    noise_res = run_retrieval_noise_experiment(k_list=k_vals, samples_per_k=samples_per_k)
    print(f" -> Retrieval Noise Completed.")

    # 3. Project 2 Visual Ambiguity Experiment
    print("\n[3/6] Running Project 2 Visual Ambiguity Stress Test...")
    visual_res = run_visual_ambiguity_experiment(samples_per_cond=samples_per_cond)
    print(f" -> Visual Ambiguity Completed.")

    # 4. Project 2 Cross-Modal Conflict Experiment
    print("\n[4/6] Running Project 2 Cross-Modal Conflict Resolution Test...")
    conflict_res = run_cross_modal_conflict_experiment(samples=conflict_samples)
    print(f" -> Conflict Detection Rate: {conflict_res['conflict_detection_rate']*100:.1f}%")

    # 5. Project 4 Attention Efficiency Experiment
    print("\n[5/6] Running Project 4 CRPA Attention Efficiency Benchmark...")
    attention_res = run_attention_efficiency_experiment(seq_length=4 if smoke_mode else 16, num_patients=4 if smoke_mode else 40)
    print(f" -> Attention Efficiency Completed.")

    # 6. Combined Multi-Failure Stress Test
    print("\n[6/6] Running Combined Med-Nexus Multi-Failure Stress Test...")
    combined_res = run_combined_med_nexus_experiment(samples=combined_samples)
    print(f" -> Med-Nexus Combined Accuracy: {combined_res['med_nexus_accuracy']*100:.2f}% (Conflict Flag Rate: {combined_res['conflict_detection_rate']*100:.1f}%)")

    # Master Aggregated Summary
    aggregated_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_evaluation": baseline_res,
        "retrieval_noise_benchmark": noise_res,
        "visual_ambiguity_benchmark": visual_res,
        "cross_modal_conflict_benchmark": conflict_res,
        "attention_efficiency_benchmark": attention_res,
        "combined_med_nexus_benchmark": combined_res
    }

    try:
        # Check if ReportGenerator exists and works
        report_gen = ReportGenerator(output_dir=output_dir)
        report_md = report_gen.save_summary_report(aggregated_summary)
        print(f"\n[SUCCESS] Master Benchmark Summary Report Saved: {report_md}")
    except Exception as e:
        print(f"\n[WARNING] Report generation failed (maybe missing module): {e}")

    print("================================================================")


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    run_all(smoke_mode=smoke)
