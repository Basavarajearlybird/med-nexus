"""
Med-Nexus Benchmark Report & Visualization Generator
Generates publication-quality charts and markdown reports for Medical Image Analysis journal submission.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


class ReportGenerator:
    """Generates benchmark plots and markdown summaries for Med-Nexus."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_retrieval_noise_results(self, k_values: list, accuracy_list: list, ece_list: list):
        """Generates line chart for Project 2 Retrieval Noise benchmark."""
        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        color = 'tab:blue'
        ax1.set_xlabel('Retrieval Distractor Count (k)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Diagnostic Accuracy (%)', color=color, fontsize=11, fontweight='bold')
        ax1.plot(k_values, [a * 100 for a in accuracy_list], color=color, marker='o', linewidth=2.5, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(50, 105)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Expected Calibration Error (ECE)', color=color, fontsize=11, fontweight='bold')
        ax2.plot(k_values, ece_list, color=color, marker='s', linestyle='--', linewidth=2.0, label='ECE')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Med-Nexus Stress Test: Diagnostic Robustness under Retrieval Noise (Project 2)', fontsize=12, fontweight='bold', pad=12)
        fig.tight_layout()
        filepath = os.path.join(self.output_dir, "retrieval_noise.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_visual_ambiguity_results(self, conditions: list, accuracy_list: list, confidence_list: list):
        """Generates bar chart for Project 2 Visual Ambiguity benchmark."""
        x = np.arange(len(conditions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        rects1 = ax.bar(x - width/2, [a * 100 for a in accuracy_list], width, label='Accuracy (%)', color='#2b5c8f')
        rects2 = ax.bar(x + width/2, [c * 100 for c in confidence_list], width, label='Mean Confidence (%)', color='#e07a5f')

        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Med-Nexus Diagnostics under Visual Ambiguity & Perturbations (Project 2)', fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in conditions], fontsize=10)
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        fig.tight_layout()
        filepath = os.path.join(self.output_dir, "visual_ambiguity.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_attention_efficiency_results(self, methods: list, latency_list: list, retention_list: list):
        """Generates bar chart for Project 4 CRPA Attention Efficiency benchmark."""
        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        color = '#457b9d'
        ax1.set_xlabel('Attention Mechanism', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Inference Latency (ms)', color=color, fontsize=11, fontweight='bold')
        ax1.bar(np.arange(len(methods)) - 0.18, latency_list, width=0.35, color=color, label='Latency (ms)')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = '#2a9d8f'
        ax2.set_ylabel('Attention Edge Retention (%)', color=color, fontsize=11, fontweight='bold')
        ax2.plot(np.arange(len(methods)) + 0.18, [r * 100 for r in retention_list], color=color, marker='D', linewidth=2.5, label='Active Edges (%)')
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_xticks(np.arange(len(methods)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], fontsize=10)

        plt.title('CRPA Attention Efficiency & Edge Gating Benchmark (Project 4)', fontsize=12, fontweight='bold', pad=12)
        fig.tight_layout()
        filepath = os.path.join(self.output_dir, "attention_efficiency.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def save_summary_report(self, results_data: dict):
        """Saves experiment output to JSON and Markdown summary report."""
        json_path = os.path.join(self.output_dir, "experiment_summary.json")
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=2)

        md_path = os.path.join(self.output_dir, "MED_NEXUS_BENCHMARK_REPORT.md")
        with open(md_path, "w") as f:
            f.write("# Med-Nexus Experimental Benchmark Report\n\n")
            f.write("**Target Journal**: Medical Image Analysis (ScienceDirect)\n")
            f.write("**Architecture**: Conflict-Resilient GIB Diagnostic Fusion & CRPA Long-Context Sequence Tracking\n\n")
            f.write("## Benchmark Results Summary\n\n")
            f.write("```json\n")
            f.write(json.dumps(results_data, indent=2))
            f.write("\n```\n")

        return md_path
