import sys
sys.path.append('core')
import os
os.environ["KMP_BLOCKTIME"] = "0"
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from logger import Logger
from argparser import ArgParser
from trainer import Trainer
from evaluate import evaluates
from utils.datasets import fetch_dataset, fetch_test_dataset
from utils.utils import setup_seed, count_parameters, count_all_parameters, build_module


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy gradscale for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def train(local_rank, args):

    args.local_rank = local_rank

    if args.local_rank == 0:
        logger = Logger(args)
        for k, v in vars(args).items():
            logger.log_debug('{}\t=\t{}'.format(k, v), "argparser")
        _print = logger.log_info
    else:
        logger = None
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    if args.distributed == 'ddp':
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.nprocs, rank=local_rank)
        torch.cuda.set_device(args.local_rank)

    train_set, val_sets, val_setnames = fetch_dataset(args)
    if args.distributed == 'ddp':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=args.seed, drop_last=False)
        train_loader = DataLoader(train_set, args.batch_size, num_workers=args.jobs, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.jobs, \
            pin_memory=True, drop_last=True, sampler=None)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use training set {} with length: bs/loader/dataset ({}/{}({})/{})".format( \
            args.stage, args.batch_size, len(train_loader), len(train_loader.dataset), len(train_set)))

    assert len(val_setnames) == len(val_sets)
    val_length_str = ""
    for val_set, name in zip(val_sets, val_setnames):
        val_length_str += "({}/{}),".format(name, len(val_set))
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use validation set: test_bs={}, name/datalength:{}".format( \
            args.test_batch_size, val_length_str))

    model = build_module("core", args.model)(args)
    if args.distributed == 'ddp':
        model.cuda(args.gpus[args.local_rank])
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model, \
            device_ids=[args.gpus[args.local_rank]])
        _print("Use DistributedDataParallel at gpu {} with find_unused_parameters:False".format( \
            args.gpus[args.local_rank]))
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
        model.cuda(args.gpus[0])
        model.train()

    loss = build_module("core.loss", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use losses: {} with weights: {}".format(args.loss, args.loss_weights))

    metric_fun = build_module("core.metric", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use metrics: {}".format(args.metric))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, \
        eps=args.epsilon)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use optimizer: {} with init lr:{}, decay:{}, epsilon:{} ".format( \
            "AdamW", args.lr, args.weight_decay, args.epsilon))

    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, \
        steps_per_epoch=len(train_loader), epochs=args.epoch)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use scheduler: {}, with epoch:{}, steps_per_epoch {}".format( \
            "OneCycleLR", args.epoch, len(train_loader)))

    scaler = GradScaler(enabled=args.mixed_precision)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use gradscaler with mixed_precision? {}".format(args.mixed_precision))

    trainer = Trainer(args, model, loss=loss, optimizer=optimizer, \
        lr_scheduler=lr_scheduler, scaler=scaler, logger=logger)
    start = 0
    if args.checkpoint != '':
        start = trainer.load(args.checkpoint, only_model=False if args.resume else True)

    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("For model {} with name {}, Parameter Count: {}(trainable)/{}(all), gpus: {}".format( \
            args.model, args.name if args.name != "" else "NoNAME", count_parameters(trainer.model), \
                count_all_parameters(trainer.model), args.gpus))
        _print("Use small? {}".format(args.small))

    setup_seed(args.seed)

    for i in range(start+1, args.epoch+1):
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print(">>> Start the {}/{} training epoch with save feq {} at stage {}".format( \
                i, args.epoch, args.save_feq, args.stage), "training")
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        trainer.run_epoch(train_loader)
        if args.local_rank == 0 and logger is not None:
            logger.summary(i)

        if i % args.eval_feq == 0:
            if args.distributed != 'ddp' or args.local_rank == 0:
                _print(">>> Run {} evaluate epoch".format(i), "training")
            scores = evaluates(args, model, val_sets, val_setnames, metric_fun, logger=logger)
            if args.local_rank == 0 and logger is not None:
                logger.write_dict(i, scores)
        if args.local_rank == 0 and i % args.save_feq == 0:
            trainer.store(args.save_path, args.name, i)

    if args.local_rank == 0:
        dist.destroy_process_group()
        _print("Destroy_process_group", 'train')

    if logger is not None:
        logger.close()


def test(local_rank, args, logger=None):

    args.local_rank = local_rank

    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    if args.distributed == 'ddp':
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.nprocs, rank=local_rank)
        torch.cuda.set_device(args.local_rank)

    assert args.checkpoint != ''

    start = time.time()
    test_sets, test_setnames = fetch_test_dataset(args)

    assert len(test_setnames) == len(test_sets)
    test_length_str = ""
    for test_set, name in zip(test_sets, test_setnames):
        test_length_str += "({}/{}),".format(name, len(test_set))
    
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use test set: test_bs={}, name/datalength:{}".format(args.test_batch_size, test_length_str), 'test')

    metric_fun = build_module("core.metric", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use metrics: {}".format(args.metric), 'test')

    model = build_module("core", args.model)(args)

    if args.checkpoint != '':
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print("Evalulate Model {} for checkpoint {}".format(args.model, args.checkpoint), 'test')
            _print("For model {} with name {}, Parameter Count: {}(trainable)/{}(all), gpus: {}".format( \
                args.model, args.name if args.name != "" else "NoNAME", count_parameters(model), \
                    count_all_parameters(model), args.gpus))

        state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        try:
            if "model" in state_dict.keys():
                state_dict = state_dict.pop("model")
            elif 'model_state_dict' in state_dict.keys():
                state_dict = state_dict.pop("model_state_dict")

            if "module." in list(state_dict.keys())[0]:
                for key in list(state_dict.keys()):
                    state_dict.update({key[7:]:state_dict.pop(key)})

            model.load_state_dict(state_dict)
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(args.checkpoint))
    else:
        raise NotImplementedError("Please set --checkpoint")

    if args.distributed == 'ddp':
        model.cuda(args.local_rank)
        model.eval()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus[local_rank]])
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
        model.cuda(args.gpus[0])
        model.eval()

    scores = evaluates(args, model, test_sets, test_setnames, metric_fun, logger=logger)

    summary_str = ""
    for key in scores.keys():
        summary_str += "({}/{}),".format(key, scores[key])

    if args.distributed != 'ddp' or args.local_rank == 0:
        dist.destroy_process_group()
        _print("Destroy_process_group", 'test')

        _print("Test complete, {}, time consuming {}/s".format(summary_str, time.time() - start), 'test')


if __name__ == "__main__":
    argparser = ArgParser()
    args = argparser.parser()
    setup_seed(args.seed)

    if args.gpus[0] == -1:
        args.gpus = [i for i in range(torch.cuda.device_count())]
    args.nprocs = len(args.gpus)

    if args.task == "train":
        if args.distributed == 'ddp':
            mp.spawn(train, nprocs=args.nprocs, args=(args, ))
        else:
            train(-1, args)
    elif args.task[:4] == "test":
        if args.distributed == 'ddp':
            mp.spawn(test, nprocs=args.nprocs, args=(args, ))
        else:
            train(-1, args)
    else:
        print("task error")
