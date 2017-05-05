#!/usr/bin/env python2.7
#

import argparse
import os
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

config, unparsed = get_config()
config.gpu_options.allow_growth = True

data_loader = get_loader(
            data_path, config.batch_size, config.input_scale_size,
            config.data_format, config.split)
trainer = Trainer(config, data_loader)
trainer.complete(config, args)