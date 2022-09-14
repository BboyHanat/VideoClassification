import os
import torch
import random
import numpy as np
from torch import nn
from logger import logger
from datetime import datetime
import torch.distributed as dist
from prettytable import PrettyTable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm, parallel
from torch.utils.tensorboard import SummaryWriter
from utils.dist_util import reduce_mean, accuracy, AverageMeter
from module.builder import network_builder, loss_builder, lr_policy_builder, \
    dataset_builder, summary_builder, optimizer_builder

__all__ = ['dist_trainer']


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def dist_trainer(local_rank, dist_num: int, config: dict):
    """

    :param local_rank:
    :param dist_num:
    :param config: distribute training parameters
    :return: 
    """
    train_cfg = config['train_cfg']
    init_seeds(local_rank + 1, cuda_deterministic=False)
    init_method = 'tcp://' + config['dist_cfg']['ip'] + ':' + str(config['dist_cfg']['port'])
    dist.init_process_group(backend='nccl',  # noqa
                            init_method=init_method,
                            world_size=dist_num,
                            rank=local_rank)

    # set different seed for each worker
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
    network_model = SyncBatchNorm.convert_sync_batchnorm(network_model).to(local_rank)
    network_model = parallel.DistributedDataParallel(network_model,
                                                     device_ids=[local_rank])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_batch_data = DataLoader(train_dataset,
                                  batch_size=train_cfg['train_batch'],
                                  shuffle=train_cfg['shuffle'],
                                  num_workers=train_cfg['num_workers'],
                                  sampler=train_sampler)
    val_batch_data = DataLoader(val_dataset,
                                batch_size=train_cfg['val_batch'],
                                shuffle=train_cfg['shuffle'],
                                num_workers=0,
                                sampler=val_sampler)

    if train_cfg['pretrained']:
        logger.info('process {} Load from: {}'.format(local_rank, train_cfg['pretrained']))
        state_dict = torch.load(train_cfg['pretrained'], map_location=torch.device('cpu'))
        network_model.load_state_dict(state_dict, strict=False)
        logger.info('process {} load finish'.format(local_rank))

    train_steps = int(len(train_dataset) / train_cfg['train_batch'] / dist_num)
    # val_steps = int(len(val_dataset) / train_cfg['val_batch'] / train_cfg['dist_num'])

    logger.info("start training")
    new_acc1 = 0.0
    for e in range(train_cfg['epoch']):
        network_model = train(network_model=network_model,
                              dataloader=train_batch_data,
                              loss_func=loss_func,
                              optimizer=optimizer,
                              lr_scheduler=lr_policy,
                              summary_writer=summary_writer,
                              local_rank=local_rank,
                              train_steps=train_steps,
                              epoch=e,
                              dist_num=dist_num)
        acc1, val_loss = validation(network_model=network_model,
                                    dataloader=val_batch_data,
                                    loss_func=loss_func,
                                    summary_writer=summary_writer,
                                    local_rank=local_rank,
                                    epoch=e,
                                    dist_num=dist_num)
        if acc1 > new_acc1:
            new_acc1 = acc1

            weight_prefix = train_cfg['weight_output_prefix'] if \
                train_cfg['weight_output_prefix'] else datetime.now().strftime('%b_%d_%H_%M')
            weight_name = weight_prefix + "_epoch_" + str(e) + "_acc_" + str(acc1) + ".pth"
            save_name = os.path.join(train_cfg['weight_output_dir'], weight_name)
            torch.save(network_model.module.state_dict(), save_name)

        if local_rank == 0:
            table = PrettyTable(['epoch', 'acc1', 'val_loss'])
            table.add_row([e, "%.4f" % acc1, "%.4f" % val_loss])
            logger.info(table)


def train(network_model: nn.Module,
          dataloader: DataLoader,
          loss_func: nn.Module,
          optimizer,
          lr_scheduler,
          summary_writer: SummaryWriter,
          local_rank: int,
          train_steps: int,
          epoch: int,
          dist_num: int):
    """

    :param network_model: network model
    :param dataloader: ddp dataloader
    :param loss_func: loss function
    :param optimizer:
    :param lr_scheduler:
    :param summary_writer:
    :param local_rank: current rank
    :param train_steps: computed equ: data-length / (batch-size * dist-num)
    :param epoch: current epoch idx
    :param dist_num:
    :return:
    """
    network_model.train()
    for ts, (x, y) in enumerate(dataloader):
        x = x.to(torch.device('cuda:{}'.format(local_rank)))
        y = y.to(torch.device('cuda:{}'.format(local_rank)))
        output = network_model(x)
        loss = loss_func(output, y)
        torch.distributed.barrier()  # noqa
        reduced_loss = reduce_mean(loss, dist_num)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if local_rank == 0:
            logger.info("epoch: {}  step: {}  loss: {}".format(epoch, ts,
                                                               reduced_loss.data.cpu().numpy()))
            summary_writer.add_scalar('train/loss_total',
                                      reduced_loss.data.cpu().numpy(),
                                      global_step=epoch * train_steps + ts)
    return network_model


def validation(network_model: nn.Module,
               dataloader: DataLoader,
               loss_func: nn.Module,
               summary_writer: SummaryWriter,
               local_rank: int,
               epoch: int,
               dist_num: int):
    """

    :param network_model: network model
    :param dataloader: ddp dataloader
    :param loss_func: loss function
    :param summary_writer:
    :param local_rank: current rank
    :param epoch: current epoch idx
    :param dist_num:
    :return:
    """
    network_model.eval()
    loss_avg = AverageMeter('loss', ':.4e')
    acc1 = AverageMeter('Accuracy', ':.4e')
    acc5 = AverageMeter('Accuracy', ':.4e')

    with torch.no_grad():
        for vs, (x, y) in enumerate(dataloader):
            x = x.to(torch.device('cuda:{}'.format(local_rank)))
            y = y.to(torch.device('cuda:{}'.format(local_rank)))
            output = network_model(x)
            loss = loss_func(output, y)
            torch.distributed.barrier()  # noqa
            reduced_loss = reduce_mean(loss, dist_num)

            acc_top1 = accuracy(output, y)[0]
            acc_top1 = acc_top1.to(torch.device('cuda:{}'.format(local_rank)))

            reduced_acc1 = reduce_mean(acc_top1, dist_num)

            loss_avg.update(reduced_loss.item(), x.size(0))
            acc1.update(reduced_acc1.item(), x.size(0))

        if local_rank == 0:
            summary_writer.add_scalar('val/loss_total',
                                      reduced_loss.data.cpu().numpy(),
                                      global_step=epoch)
            summary_writer.add_scalar('val/acc1',
                                      acc1.avg,
                                      global_step=epoch)

        return acc1.avg, reduced_loss.avg
