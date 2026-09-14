"""Read-only inference audit of every module input/output, including residuals."""
import torch
from contextlib import contextmanager


class FiniteAudit:
    def __init__(self, model):
        self.stats = {}
        self.enabled = True
        self.handles = []
        for name, module in model.named_modules():
            self.handles.append(module.register_forward_pre_hook(
                lambda m, inputs, n=name: self.observe(n+'.input', inputs)))
            self.handles.append(module.register_forward_hook(
                lambda m, inputs, output, n=name: self.observe(n+'.output', output)))

    def observe(self, name, value):
        if not self.enabled:
            return
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
            rows = int(value.shape[0]) if value.ndim else 1
            if name not in self.stats:
                self.stats[name] = [bad, peak, rows]
            else:
                self.stats[name][0] += bad
                self.stats[name][1] = torch.maximum(self.stats[name][1], peak)
                self.stats[name][2] += rows

    def result(self):
        modules = {n: dict(nonfinite_values=int(v[0]), finite_absmax=float(v[1]), observed_rows=v[2])
                   for n, v in self.stats.items()}
        return dict(nonfinite_values=sum(v['nonfinite_values'] for v in modules.values()),
                    boundaries=modules)

    @contextmanager
    def paused(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def close(self):
        for handle in self.handles:
            handle.remove()
