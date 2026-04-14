import os
import json
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from ..core import get_activation, get_dataset, ProbeNetwork
from ..logic import Trainer, TrainingResult
from ..visualizer import Visualizer

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    activation: str = "Sigmoid"
    custom_expr: Optional[str] = None
    loss_type: str = "mse"
    n_layers: int = 5
    hidden_dim: int = 32
    dataset: str = "xor"
    epochs: int = 100
    lr: float = 0.01
    label: str = "default"

class Experiment:
    """
    Orchestrates the creation, training, and visualization of an experiment.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_base = "results"
        self.output_dir = os.path.join(self.output_base, config.label)
        self.visualizer = Visualizer(self.output_dir)

    def run(self) -> TrainingResult:
        """Runs the experiment pipeline."""
        print(f"\n[Experiment] Starting: {self.config.label}")
        print(f" - Activation: {self.config.activation}")
        if self.config.custom_expr:
            print(f" - Custom Expr: {self.config.custom_expr}")
        print(f" - Layers: {self.config.n_layers}, Dataset: {self.config.dataset}")
        
        # 1. Dataset
        X, y = get_dataset(self.config.dataset)
        input_dim = X.shape[1]
        output_dim = y.shape[1] if len(y.shape) > 1 else (y.max().item() + 1 if y.dtype == torch.long else 1)
        
        # 2. Model
        act_fn = get_activation(self.config.activation, self.config.custom_expr)
        model = ProbeNetwork(
            input_dim=input_dim, 
            n_layers=self.config.n_layers, 
            hidden_dim=self.config.hidden_dim, 
            output_dim=output_dim, 
            activation_fn=act_fn
        )
        
        # 3. Training
        trainer = Trainer(model, lr=self.config.lr, loss_type=self.config.loss_type)
        result = trainer.train(X, y, epochs=self.config.epochs)
        
        # 4. Result Processing
        result.config = asdict(self.config)
        self._save_metrics(result)
        self.visualizer.generate_all_plots(result, label=self.config.label)
        
        self._print_summary(result)
        print(f"[Experiment] Completed. Results saved to: {self.output_dir}")
        return result

    def _print_summary(self, result: TrainingResult):
        """Prints a human-readable summary of the experiment results."""
        print("\n" + "-"*45)
        print(f"RESULT SUMMARY: {self.config.label}")
        print("-"*45)
        print(f"Final Loss: {result.final_loss:.6f}")
        
        layers = sorted(result.gradient_norms.keys())
        if not layers:
            return

        # Calculate Gradient Health (First layer vs Last layer avg norm)
        first_layer = layers[0]
        last_layer = layers[-1]
        
        avg_first = np.mean(result.gradient_norms[first_layer])
        avg_last = np.mean(result.gradient_norms[last_layer])
        
        ratio = avg_first / (avg_last + 1e-12)
        
        print(f"Layer 0 Avg Grad: {avg_first:.2e}")
        print(f"Layer {last_layer} Avg Grad: {avg_last:.2e}")
        print(f"Health Ratio (L0/L{last_layer}): {ratio:.2e}")
        
        print("\nDIAGNOSIS:")
        if ratio < 1e-5:
            print("[CRITICAL] Severe Vanishing Gradient detected!")
            print("   Training in early layers has practically stopped.")
        elif ratio < 1e-3:
            print("[WARNING] Significant Vanishing Gradient observed.")
        elif ratio > 1e2:
            print("[WARNING] Exploding Gradient detected!")
        else:
            print("[HEALTHY] Gradients are flowing well across the network.")
            
        print("-"*45)

    def _save_metrics(self, result: TrainingResult):
        """Saves numerical metrics and config to JSON."""
        metrics = {
            "final_loss": result.final_loss,
            "avg_grad_norms": {str(k): float(np.mean(v)) for k, v in result.gradient_norms.items()},
            "config": result.config
        }
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

def compare(configs: List[ExperimentConfig]):
    """Runs multiple configurations for easy comparison."""
    results = []
    for cfg in configs:
        results.append(Experiment(cfg).run())
    return results

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(
        activation="Sigmoid", 
        n_layers=6, 
        epochs=50, 
        label="sigmoid_test"
    )
    Experiment(config).run()
