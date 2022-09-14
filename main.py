import yaml
import torch
import random
import argparse
import numpy as np
import torch.multiprocessing as mp

from tools.trainer import dist_trainer


def init_config(cfg):
    with open(cfg, 'r') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)
    return conf


def args_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-cfg', '--config', default='configs/dist_slowfast_train_config.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    config = init_config(cfg=args.config)
    dist_cfg = config['dist_cfg']
    mp.spawn(dist_trainer, nprocs=dist_cfg['dist_num'],
             args=(dist_cfg['dist_num'], config))


if __name__ == "__main__":
    main()

