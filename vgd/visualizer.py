import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from .logic.trainer import TrainingResult
from typing import Optional, List

class Visualizer:
    """
    Handles plotting and saving results from experiments.
    """
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss(self, result: TrainingResult, label: str = "Training"):
        plt.figure(figsize=(10, 5))
        plt.plot(result.loss_history, label=f"{label} Loss", linewidth=2)
        plt.title(f"Loss Curve", fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"), dpi=200)
        plt.close()

    def plot_gradient_flow(self, result: TrainingResult, label: str = "Training"):
        plt.figure(figsize=(12, 6))
        layers = sorted(result.gradient_norms.keys())
        
        # Plot average, min, and max gradient norms across the whole run for each layer
        avg_norms = [np.mean(result.gradient_norms[i]) for i in layers]
        
        plt.plot(layers, avg_norms, marker='o', markersize=8, linewidth=2, label="Avg Gradient Norm")
        
        plt.title(f"Gradient Flow across Layers", fontsize=14)
        plt.xlabel("Linear Layer Index")
        plt.ylabel("L2 Norm of Gradient (Log Scale)")
        plt.yscale('log')
        plt.xticks(layers)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(os.path.join(self.output_dir, "gradient_flow.png"), dpi=200)
        plt.close()

    def plot_activation_distributions(self, result: TrainingResult, label: str = "Training"):
        layers = sorted(result.activation_histograms.keys())
        n_layers = len(layers)
        
        if n_layers == 0:
            return

        fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4), sharey=True)
        if n_layers == 1:
            axes = [axes]
            
        for i, layer_idx in enumerate(layers):
            data = result.activation_histograms[layer_idx].numpy().flatten()
            data_range = np.max(data) - np.min(data)
            if data.size > 0 and data_range > 1e-4:
                axes[i].hist(data, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            else:
                val = np.mean(data)
                axes[i].text(0.5, 0.5, f"Near-Constant\n({val:.4f})", 
                             transform=axes[i].transAxes, ha='center', va='center', fontsize=10)
            axes[i].set_title(f"Layer {layer_idx}")
            axes[i].set_xlabel("Activation Value")
            if i == 0:
                axes[i].set_ylabel("Frequency")
        
        plt.suptitle(f"Activation Distributions (Final Epoch)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "activation_distributions.png"), dpi=200)
        plt.close()

    def plot_weight_updates(self, result: TrainingResult, label: str = "Training"):
        plt.figure(figsize=(10, 5))
        layers = sorted(result.weight_update_magnitudes.keys())
        for layer_idx in layers:
            plt.plot(result.weight_update_magnitudes[layer_idx], label=f"Layer {layer_idx}")
            
        plt.title(f"Weight Update Magnitude per Epoch", fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("L2 Norm of Weight Change")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "weight_updates.png"), dpi=200)
        plt.close()

    def generate_all_plots(self, result: TrainingResult, label: str = "unnamed_run"):
        """Generates all standard plots for a result."""
        self.plot_loss(result, label)
        self.plot_gradient_flow(result, label)
        self.plot_activation_distributions(result, label)
        self.plot_weight_updates(result, label)
