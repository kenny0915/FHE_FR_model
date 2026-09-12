"""Read-only inference audit of every module input/output, including residuals."""
import torch


class FiniteAudit:
    def __init__(self, model):
        self.stats = {}
        self.handles = []
        for name, module in model.named_modules():
            self.handles.append(module.register_forward_pre_hook(
                lambda m, inputs, n=name: self.observe(n+'.input', inputs)))
            self.handles.append(module.register_forward_hook(
                lambda m, inputs, output, n=name: self.observe(n+'.output', output)))

    def observe(self, name, value):
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.observe(name+f'[{index}]', item)
            return
        if not torch.is_tensor(value):
            return
        with torch.no_grad():
            finite = torch.isfinite(value)
            bad = (~finite).sum()
            peak = torch.where(finite, value.abs(), 0).amax().double()
            if name not in self.stats:
                self.stats[name] = [bad, peak]
            else:
                self.stats[name][0] += bad
                self.stats[name][1] = torch.maximum(self.stats[name][1], peak)

    def result(self):
        modules = {n: dict(nonfinite_values=int(v[0]), finite_absmax=float(v[1]))
                   for n, v in self.stats.items()}
        return dict(nonfinite_values=sum(v['nonfinite_values'] for v in modules.values()),
                    boundaries=modules)

    def close(self):
        for handle in self.handles:
            handle.remove()
