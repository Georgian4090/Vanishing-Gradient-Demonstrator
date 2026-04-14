import torch
import torch.nn as nn
from typing import List, Dict, Any

class ProbeNetwork(nn.Module):
    """
    A configurable MLP that collects gradients and activations during training.
    
    Attributes:
        gradient_norms (Dict[int, List[float]]): Maps layer index to list of gradient norms.
        activations (Dict[int, List[torch.Tensor]]): Maps layer index to last recorded activations.
    """
    def __init__(self, input_dim: int, n_layers: int, hidden_dim: int, output_dim: int, activation_fn: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build layers
        current_dim = input_dim
        for i in range(n_layers):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(activation_fn)
            current_dim = hidden_dim
        
        self.layers.append(nn.Linear(current_dim, output_dim))
        
        # Storage for analytics
        self.gradient_norms = {i: [] for i, l in enumerate(self.layers) if isinstance(l, nn.Linear)}
        self.last_activations = {i: None for i, l in enumerate(self.layers) if isinstance(l, nn.Linear)}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks for data collection."""
        def get_activation_hook(idx):
            def hook(module, input, output):
                self.last_activations[idx] = output.detach().cpu()
            return hook

        def get_gradient_hook(idx):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    norm = grad_output[0].norm().item()
                    self.gradient_norms[idx].append(norm)
            return hook

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                self._hooks.append(layer.register_forward_hook(get_activation_hook(i)))
                self._hooks.append(layer.register_full_backward_hook(get_gradient_hook(i)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def clear_logs(self):
        """Clears the collected logs."""
        for k in self.gradient_norms:
            self.gradient_norms[k] = []
        for k in self.last_activations:
            self.last_activations[k] = None

    def __del__(self):
        """Ensure hooks are removed."""
        for h in self._hooks:
            h.remove()

if __name__ == "__main__":
    from .activations import get_activation
    act = get_activation("sigmoid")
    model = ProbeNetwork(2, 3, 16, 1, act)
    x = torch.randn(10, 2)
    y = model(x)
    y.sum().backward()
    print(f"Gradient norms per layer: { {k: len(v) for k, v in model.gradient_norms.items()} }")
