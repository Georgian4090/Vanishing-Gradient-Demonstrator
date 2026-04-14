import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from typing import Callable, Optional, Dict, Any

class CustomActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, f, df):
        ctx.df = df
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        output_np = np.asarray(f(input_np))
        return torch.from_numpy(output_np).to(input.device).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        df = ctx.df
        input_np = input.detach().cpu().numpy()
        grad_np = np.asarray(df(input_np))
        grad_input = torch.from_numpy(grad_np).to(input.device).to(input.dtype)
        return grad_output * grad_input, None, None

class CustomActivation(nn.Module):
    """
    A PyTorch module for custom activation functions defined by strings.
    Uses SymPy for symbolic differentiation and wraps it in a torch.autograd.Function.
    """
    def __init__(self, expr_str: str):
        super().__init__()
        self.expr_str = expr_str
        x = sp.symbols('x', real=True)
        try:
            # Common functions for convenience
            context = {
                'exp': sp.exp,
                'sin': sp.sin,
                'cos': sp.cos,
                'log': sp.log,
                'abs': sp.Abs,
                'tanh': sp.tanh,
                'sigmoid': lambda v: 1 / (1 + sp.exp(-v))
            }
            self.expr = sp.sympify(expr_str, locals=context)
            self.deriv_expr = sp.diff(self.expr, x).doit()
            
            # Lambdify for fast numerical evaluation
            self.f_lambdified = sp.lambdify(x, self.expr, 'numpy')
            self.df_lambdified = sp.lambdify(x, self.deriv_expr, 'numpy')
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {expr_str}. Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CustomActivationFunction.apply(x, self.f_lambdified, self.df_lambdified)

def get_activation(name: str, custom_expr: Optional[str] = None) -> nn.Module:
    """
    Factory function for activation functions.
    """
    name_clean = name.lower().replace(" ", "").replace("_", "")
    
    if name_clean == "sigmoid":
        return nn.Sigmoid()
    elif name_clean == "tanh":
        return nn.Tanh()
    elif name_clean == "relu":
        return nn.ReLU()
    elif name_clean == "leakyrelu":
        return nn.LeakyReLU()
    elif name_clean == "elu":
        return nn.ELU()
    elif name_clean == "swish":
        return nn.SiLU()
    elif name_clean == "custom":
        if not custom_expr:
            raise ValueError("Custom activation requires 'custom_expr' string.")
        return CustomActivation(custom_expr)
    else:
        raise ValueError(f"Activation '{name}' not supported. Choose from: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, Custom.")

if __name__ == "__main__":
    # Quick test
    try:
        act = get_activation("custom", "x**2 + 2*x + 1")
        test_input = torch.tensor([1.0, 2.0], requires_grad=True)
        out = act(test_input)
        print(f"Outcome: {out}")
        out.sum().backward()
        print(f"Gradient: {test_input.grad}") # Should be 2x + 2 -> [4, 6]
    except Exception as e:
        print(f"Test failed: {e}")
