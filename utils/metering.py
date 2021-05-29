import time
import torch


class Timer:

    def __init__(self, gpu: bool):
        self.gpu = gpu
        self.start = None
        self.end = None
        self.reset()

    def reset(self):
        self._gpu_reset() if self.gpu else self._reset()

    def elapsed(self) -> float:
        return self._gpu_elapsed() if self.gpu else self._elapsed()

    def _gpu_reset(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def _gpu_elapsed(self):
        self.end = torch.cuda.Event(enable_timing=True)
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) / 1000.0

    def _reset(self):
        self.start = time.perf_counter()

    def _elapsed(self):
        self.end = time.perf_counter()
        return self.end - self.start


class RunningAverage:
    def __init__(self, name: str, fmt: str):
        self.name = name
        self.fmt = fmt
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.last = 0

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.last = val

    def __str__(self):
        fmtstr = '{} (avg = {' + self.fmt + '}, last = {' + self.fmt + '})'
        return fmtstr.format(self.name, self.avg, self.last)
