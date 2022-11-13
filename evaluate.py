import sys
sys.path.append('core')
import os
os.environ["KMP_BLOCKTIME"] = "0"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils.utils import InputPadder


def reduce_list(lists, nprocs):
    new_lists = {}
    for key, value in lists.items():
        rt = value.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        new_lists[key] = rt.item()
    return new_lists


def reduce_tensor(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def evaluates(args, model, datasets, names, metric_fun, logger=None):
    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    metrics = {}
    for val_set, name in zip(datasets, names):
        if args.distributed == 'ddp':
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, \
                seed=args.seed, drop_last=False)
        else:
            val_sampler = None
        val_loader = DataLoader(val_set, args.test_batch_size, num_workers=args.jobs, sampler=val_sampler)
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print(">>> For evaluate {}, use length (bs/loader/set): ({}/{}/{})".format( \
                name, args.test_batch_size, len(val_loader), len(val_set)), "evaluates")
        metric = evaluate(args, model, val_loader, name, metric_fun, logger=logger)

        for key, values in metric.items():
            new_key = "val_{}/{}".format(name, key)
            assert new_key not in metrics
            metrics[new_key] = values

    return metrics


def evaluate(args, model, dataloader, name, metric_fun, logger=None):
    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    start = time.time()
    model.eval()

    metric_fun.clear()

    if args.distributed != 'ddp' or args.local_rank == 0:
        bar = tqdm(total=len(dataloader), position=0, leave=True)

    for index, batch in enumerate(dataloader):

        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].cuda(args.gpus[args.local_rank] \
                    if args.local_rank != -1 else 0, non_blocking=True)

        padder = InputPadder(batch['image1'].shape, div=args.pad)
        pad_batch = padder.pad_batch(batch)

        torch.cuda.synchronize()
        tm = time.time()

        with torch.no_grad():
            output = model(pad_batch, iters=args.iters)

        torch.cuda.synchronize()
        elapsed = time.time() - tm

        output['flow_pred'] = padder.unpad(output['flow_final'])
        if args.isbi and 'flow_final_bw' in output.keys():
            output['flow_pred_bw'] = padder.unpad(output['flow_final_bw'])

        if 'image1_valid' in batch.keys():
            output['flow_pred'][batch['image1_valid'].repeat(1, 2, 1, 1) < 0.5] = 0

        metric_each = metric_fun.calculate(output, batch, name)

        if args.distributed == 'ddp':
            torch.distributed.barrier()
            reduced_metric_each = reduce_list(metric_each, args.nprocs)
        else:
            reduced_metric_each = metric_each

        reduced_metric_each.update({'time': elapsed})

        if args.distributed != 'ddp' or args.local_rank == 0:
            metric_fun.push(reduced_metric_each)

        if args.distributed != 'ddp' or args.local_rank == 0:
            if 'masked_epe' in metric_each.keys():
                bar.set_description("{}/{}[{}:{}], time:{:8.6f}, epe:{:8.6f}, masked_epe:{:8.6f}".format(index * len(batch['basename']), \
                    len(dataloader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe'], metric_each['masked_epe']))
            else:
                bar.set_description("{}/{}[{}:{}],time:{:8.6f}, epe:{:8.6f}".format(index * len(batch['basename']), \
                    len(dataloader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe']))
            bar.update(1)

    if args.distributed != 'ddp' or args.local_rank == 0:
        bar.close()
    metrics_str, all_metrics = metric_fun.summary()
    metric_fun.clear()

    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("<<< In {} eval: {} (100X F1), with time {}s.".format(name, metrics_str, time.time() - start), "evaluate")

    model.train()
    return all_metrics
