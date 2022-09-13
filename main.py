import yaml
import argparse
import torch.multiprocessing as mp
from tools.trainer import dist_trainer
from module.builder import network_builder, loss_builder, lr_policy_builder, \
    dataset_builder, summary_builder, optimizer_builder


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

    network_model = network_builder(config['network_cfg']['network_name'],
                                    **config['network_cfg']['model_param'])

    train_dataset = dataset_builder(config['dataset_cfg']['dataset_name'],
                                    mode='train',
                                    **config['dataset_cfg']['dataset_param'])

    val_dataset = dataset_builder(config['dataset_cfg']['dataset_name'],
                                  mode='val',
                                  **config['dataset_cfg']['dataset_param'])

    loss_func = loss_builder(config['loss_cfg']['loss_name'],
                             config['loss_cfg']['loss_weights'],
                             **config['loss_cfg']['loss_param'])

    optimizer = optimizer_builder(config['optimizer_cfg']['optimizer_name'],
                                  network_model.parameters(),
                                  **config['optimizer_cfg']['optimizer_param'])

    lr_policy = lr_policy_builder(config['lr_cfg']['lr_name'],
                                  optim=optimizer,
                                  warmup_step=config['lr_cfg']['warmup_step'],
                                  warmup_lr=config['lr_cfg']['warmup_lr'],
                                  base_lr=config['lr_cfg']['base_lr'],
                                  **config['lr_cfg']['lr_param'])

    summary_writer = summary_builder(**config['summary_cfg'])

    config['train_cfg'].update({
        'network_model': network_model,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'loss_func': loss_func,
        'optimizer': optimizer,
        'lr_policy': lr_policy,
        'summary_writer': summary_writer})

    mp.spawn(dist_trainer,
             nprocs=config['train_cfg']['dist_num'],
             args=(config['train_cfg'],))


if __name__ == "__main__":
    main()
