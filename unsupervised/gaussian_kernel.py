import torch

class EigenDecompositionV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, M):
        ut, s, u = torch.svd(M)  # s in a descending sequence.
        s = torch.clamp(s, min=1e-5)
        ctx.save_for_backward(M, u, s)
        return s, u

    @staticmethod
    def backward(ctx, dL_ds, dL_du):
        M, u, s = ctx.saved_tensors
        K_t = EigenDecompositionV2.geometric_approximation(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + torch.diag(dL_ds)).mm(u_t)
        return dL_dM

    @staticmethod
    def geometric_approximation(s):
        dtype = s.dtype
        I = torch.eye(s.shape[0], device=s.device).type(dtype)
        p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
        p = torch.where(p < 1., p, 1. / p)
        a1 = s.repeat(s.shape[0], 1).t()
        a1_t = a1.t()
        a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
        a1 *= torch.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
        p_app = torch.ones_like(p)
        p_hat = torch.ones_like(p)
        for i in range(9):
            p_hat = p_hat * p
            p_app += p_hat
        a1 = a1 * p_app
        return a1


class GaussianKernel:

    def __init__(self, max_samples: int = -1, kernel_lambda: float = 1.0, add_regularization: bool = False):
        self._max_samples = max_samples
        self._kernel_lambda = kernel_lambda
        self._add_regularization = add_regularization

    def compute_kernel(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_reduced = (A * A).sum(dim=-1).reshape(-1, 1)  # column vector (num_samples, 1)
        B_reduced = (B * B).sum(dim=-1).reshape(1, -1)  # row vector (1, num_samples)
        AB = A @ B.T  # (num_samples, num_samples)
        N = A_reduced + B_reduced - 2 * AB
        return torch.exp(- N / (N.mean() * self._kernel_lambda))

    def compute_d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._max_samples > 0 and x.size(0) > self._max_samples:
            samples_to_take = torch.randperm(x.size(0))[:self._max_samples]
            x = x[samples_to_take]
            y = y[samples_to_take]

        K = self.compute_kernel(x, x)
        lambda_, U = EigenDecompositionV2.apply(K)
        P = U @ torch.diag(
            lambda_.clamp(min=0.0, max=1.0)) @ U.T  # Projection matrix, approximation due to numeric instabilities
        y_perp = y - P.to(y.device) @ y
        d_ = ((y_perp ** 2 + 1e-5).mean(0) ** 0.5).mean()
        if self._add_regularization:
            d_ += (0.5 - K.mean())**2  # Keep the K with 0.5 average

        return d_
