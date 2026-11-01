"""Persistent progress reporting without per-batch terminal redraws."""
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def limit_memory_from_env():
    limit = int(os.environ.get('N4D_MEMORY_LIMIT_MB', '0'))
    if limit:
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        torch.cuda.set_per_process_memory_fraction(limit / total_mb, 0)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


class Progress:
    def __init__(self, iterable, *, stage, epoch, epochs, every, save_dir):
        self.iterable = iterable
        self.stage, self.epoch, self.epochs = stage, epoch, epochs
        self.every, self.save_dir = every, Path(save_dir)
        self.started = time.time()
        self.batch = 0
        self.last = {}

    def __iter__(self):
        for self.batch, value in enumerate(self.iterable, 1):
            yield value

    def set_postfix(self, values):
        self.last = values
        if self.batch != 1 and self.batch % self.every and self.batch != len(self.iterable):
            return
        status = {'stage': self.stage, 'epoch': self.epoch, 'epochs': self.epochs,
                  'batch': self.batch, 'batches': len(self.iterable), 'metrics': values,
                  'elapsed_epoch_seconds': time.time() - self.started,
                  'pid': os.getpid(), 'updated_unix': time.time()}
        atomic_json(self.save_dir.parent / 'progress.json', status)
        print('PROGRESS ' + json.dumps(status), flush=True)
