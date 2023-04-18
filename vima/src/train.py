# training code for VIMA

from __future__ import print_function
import sys
import os
import time
import argparse
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from vima.data.dataset import VIMADataset
from vima.data.collate import collate_fn
from vima.models.vima import VIMA
from vima.utils.util import AverageMeter, save_checkpoint, load_checkpoint
from vima.utils.logger import setup_logger
from vima.utils.config import update_config
from vima.utils.eval import evaluate
from vima.utils.visualize import visualize

def parse_args():
    parser = argparse.ArgumentParser(description='Train VIMA')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpu', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--resume', help='path to latest checkpoint (default: none)', default=None, type=str)
    parser.add_argument('--evaluate', help='evaluate model on validation set', action='store_true')
    parser.add_argument('--visualize', help='visualize model on validation set', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.cfg is not None:
        update_config(args.cfg)

    # set up logger
    logger, final_output_dir, tb_log_dir = setup_logger(config)
    logger.info('config:{}\n'.format(config))

    # set up tensorboard
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # set up checkpoint dir
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    checkpoint_dir = os.path.join(final_output_dir, 'model')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # set up gpu device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device: {}\n'.format(device))

    #