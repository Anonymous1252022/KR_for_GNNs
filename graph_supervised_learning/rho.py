import torch
from typing import Tuple

'''
    rho via RKHS:
'''
class ED(torch.autograd.Function):
    '''
        There is numerical instability in gradient computation of eigendecomposition:
            lambda_, U = torch.linalg.eigh(K)
        Thus we provide numerically stable backward.
    '''
    @staticmethod
    def forward(ctx, K:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lambda_, U = torch.linalg.eigh(K)
        ctx.save_for_backward(lambda_, U)
        return lambda_, U

    @staticmethod
    def backward(ctx, lambda_grad: torch.Tensor, U_grad: torch.Tensor) -> torch.Tensor:
        lambda_, U = ctx.saved_tensors
        I = torch.eye(lambda_.shape[0], device=lambda_.device)
        tmp = lambda_.reshape(-1, 1) - lambda_.reshape(1, -1) + I
        eps = 1e-5
        tmp = tmp + (tmp == 0) * eps # prevent nans
        K_tilde = 1/tmp - I
        # K_tilde = K_tilde.clamp(-1/eps, 1/eps) # prevent nans
        return U @ (K_tilde.T * (U.T @ U_grad) + torch.diag(lambda_grad)) @ U.T


def kernel(A: torch.Tensor, B: torch.Tensor, lambda_:float=1.0) -> torch.Tensor:
    '''
        Gaussian kernel, which is universal according:
            https://jmlr.csail.mit.edu/papers/volume7/micchelli06a/micchelli06a.pdf

        A: tensor of shape (num_samples, dim)
        B: tensor of shape (num_samples, dim)
    '''
    A_reduced = (A * A).sum(dim=-1).reshape(-1, 1)  # column vector (num_samples, 1)
    B_reduced = (B * B).sum(dim=-1).reshape(1, -1)  # row vector (1, num_samples)
    AB = A @ B.T # (num_samples, num_samples)
    N = A_reduced + B_reduced - 2 * AB
    return torch.exp(- N / (N.mean() * lambda_))


def compute_rho(X:torch.Tensor, Y:torch.Tensor, lambda_:float=1.0) -> float:
    '''
        Check if Y could be learned from X.

        X: tensor of shape (num_samples, dim_1)
        Y: tensor of shape (num_samples, dim_2)
    '''
    K = kernel(X, X, lambda_)
    l, U = ED.apply(K)
    P = U @ torch.diag(l.clamp(min=0.0, max=1.0)) @ U.T
    Y_perp = Y - P @ Y

    # the last sum is according dim_2 which could be > 1.
    # rho = ((Y_perp ** 2).mean(0) ** 0.5).mean() # This computation of rho is unstable during backprop of sqrt (**0.5).
    rho = (Y_perp ** 2).mean() # Stable version.
    return rho







