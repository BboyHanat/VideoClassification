import torch
from module.networks import *
from module.datasets import *
from module.losses import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader


def network_builder(model_name: str, **kwargs):
    """

    :param model_name:
    :param kwargs:
    :return: network model
    """
    model = eval(model_name)(**kwargs)
    return model


def loss_builder(loss_name: tuple,
                 loss_weights: tuple,
                 **kwargs):
    """

    :param loss_name:
    :param loss_weights:
    :param kwargs:
    :return:
    """
    if len(loss_name) == 1:
        loss_func = eval(loss_name[0])(loss_weights[0], **kwargs)
    else:
        loss_list = list()
        for idx, ln in enumerate(loss_name):
            loss_list.append(eval(ln)(loss_weights[idx], **kwargs))
        loss_func = ChainedLoss(losses=tuple(loss_list))
    return loss_func


def dataset_builder(dataset_name: str,
                    mode='train',
                    **kwargs):
    """

    :param dataset_name:
    :param mode:
    :param kwargs:
    :return:
    """

    assert mode in ['train', 'val'], "There only have 'train' and 'val' mode."
    dataset = eval(dataset_name)(mode=mode, **kwargs)
    return dataset


def optimizer_builder(optimizer_name: str,
                      model_params,
                      **kwargs):
    optim = eval(optimizer_name)(model_params, **kwargs)
    return optim


def lr_policy_builder(lr_name,
                      optim,
                      warmup_step=0,
                      warmup_lr=0.0000001,
                      base_lr=0.0001,
                      **kwargs):
    warmup_policy = None
    optim_lr = optim.param_groups[0]['lr']  # noqa
    if warmup_step > 0 and optim_lr == warmup_lr and base_lr < warmup_lr:
        interval = (base_lr - warmup_lr) / warmup_step

        def lr_lambda(step):
            new_lr = warmup_lr + step * interval
            return new_lr
        warmup_policy = LambdaLR(optim, lr_lambda, last_epoch=warmup_step)
    lr_policy = eval(lr_name)
    lr_policy = lr_policy(optim, **kwargs)
    if warmup_policy:
        lr_policy = SequentialLR(optim, schedulers=[warmup_policy, lr_policy], milestones=[warmup_step])
    return lr_policy


def summary_builder(summary_path):
    """

    :param summary_path:
    :return:
    """
    summary = SummaryWriter(log_dir=summary_path)
    return summary
