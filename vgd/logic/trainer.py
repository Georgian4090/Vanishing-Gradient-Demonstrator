import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class TrainingResult:
    """Holds the metrics collected during a training run."""
    loss_history: List[float]
    gradient_norms: Dict[int, List[float]]  # layer_idx -> [norm_epoch1, ...]
    activation_histograms: Dict[int, torch.Tensor] # layer_idx -> Last epoch activations
    weight_update_magnitudes: Dict[int, List[float]]
    final_loss: float
    config: Dict[str, Any] = field(default_factory=dict)

class Trainer:
    """
    Handles the training loop and metric accumulation.
    """
    def __init__(self, model: nn.Module, lr: float = 0.01, loss_type: str = "mse"):
        self.model = model
        self.lr = lr
        
        if loss_type.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type.lower() in ["cross_entropy", "bce"]:
            # Note: For BCE, model should output sigmoid/logit. 
            # We assume the model architecture handles the final activation appropriately or use logit-based loss.
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> TrainingResult:
        loss_history = []
        weight_update_magnitudes = {i: [] for i, l in enumerate(self.model.layers) if isinstance(l, nn.Linear)}
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Snapshot weights to calculate update magnitude
            old_weights = {i: l.weight.data.clone() for i, l in enumerate(self.model.layers) if isinstance(l, nn.Linear)}
            
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            # Calculate weight update magnitude (L2 norm of diff)
            with torch.no_grad():
                for i, l in enumerate(self.model.layers):
                    if isinstance(l, nn.Linear):
                        diff = (l.weight.data - old_weights[i]).norm().item()
                        weight_update_magnitudes[i].append(diff)
                        
        return TrainingResult(
            loss_history=loss_history,
            gradient_norms=self.model.gradient_norms,
            activation_histograms={i: act for i, act in self.model.last_activations.items() if act is not None},
            weight_update_magnitudes=weight_update_magnitudes,
            final_loss=loss_history[-1]
        )

if __name__ == "__main__":
    from ..core import ProbeNetwork, get_activation, get_dataset
    
    X, y = get_dataset("xor", n_samples=100)
    model = ProbeNetwork(2, 2, 8, 1, get_activation("sigmoid"))
    trainer = Trainer(model, lr=0.1)
    result = trainer.train(X, y, epochs=10)
    print(f"Final loss: {result.final_loss}")
    print(f"Gradient norms recorded for layers: {list(result.gradient_norms.keys())}")
