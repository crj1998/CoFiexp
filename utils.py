import logging
import os, sys, time, math
import random
import numpy as np
from datetime import datetime

import torch

def colorstr(*inputs):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
    string = str(string)
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def get_logger(args, name, level=logging.INFO, fmt="%(asctime)s [%(levelname)s @ %(name)s] %(message)s", rank=-1):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # unlike the root logger, a custom logger can’t be configured using basicConfig()
    # logging.basicConfig(
    #     filename = os.path.join(args.out, f"{time_str() if level==logging.INFO else 'dev'}_{rank}.log"),
    #     format = fmt, datefmt="%Y-%m-%d %H:%M:%S", level = level, filemode="w"
    # )

    # output to file
    file_handler = logging.FileHandler(os.path.join(args.out, f"{'dev' if 'dev' in args.out else time_str()}_{rank}.log"), mode='w')
    # file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # output to console
    stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    return logger


def setup_seed(seed):
    """ set seed for the whole program for removing randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, val, num=1):
        self.total += val * num
        self.count += num

    def item(self):
        return self.total/self.count