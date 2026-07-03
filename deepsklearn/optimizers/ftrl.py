import torch
class FTRL(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.1, beta=1.0, l1=1, l2=0, epsilon=1e-8):
        default_parameters = dict(alpha=alpha,
                                  beta=beta,
                                  l1=l1,
                                  l2=l2,
                                  epsilon=epsilon)
        super().__init__(params, default_parameters)

    @torch.no_grad()
    def step(self):
        params = self.param_groups[0]["params"]
        self.alpha = self.param_groups[0]["alpha"]
        self.beta = self.param_groups[0]["beta"]
        self.l1 = self.param_groups[0]["l1"]
        self.l2 = self.param_groups[0]["l2"]
        self.epsilon = self.param_groups[0]["epsilon"]

        for param in params:
            state = self.state[param]
            if len(state) == 0:
                state = self.state[param]
                state["n"] = torch.zeros_like(param)
                state["z"] = torch.zeros_like(param)
            if param.grad != None and param.requires_grad == True:
                if param.grad.is_sparse == True:
                    raise ValueError("not support sparse gradient")
                g = param.grad
                n = state["n"]
                z = state["z"]
                old_n = torch.sqrt(n)
                n += g ** 2
                sigma = (torch.sqrt(n) - old_n) / self.alpha
                z += g - sigma * param
                denom = (self.beta + torch.sqrt(n)) / self.alpha + self.epsilon
                w = -(z - torch.sign(z) * self.l1) / denom
                mask = torch.abs(z) > self.l1
                w = w * mask
                state["n"] = n
                state["z"] = z
                param.copy_(w)