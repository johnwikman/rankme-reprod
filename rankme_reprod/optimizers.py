import torch

class LARS(torch.optim.Optimizer):
    """
    This code is taken from
    https://github.com/facebookresearch/vissl/blob/main/vissl/optimizers/lars.py
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = False,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eta": eta,
            "exclude": exclude_bias_and_norm,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _exclude_bias_and_norm(p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                dp = dp.add(p, alpha=g["weight_decay"])

                if not g["exclude"] or not self._exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)

                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
