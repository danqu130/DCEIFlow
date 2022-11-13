import sys
sys.path.append('core')
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.utils import ensure_folder


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


class Trainer:
    def __init__(self, args, model, loss=None, optimizer=None, logger=None, lr_scheduler=None, scaler=None):
        self.args = args
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler

        if self.logger is None:
            def print_line(line, subname=None):
                if self.args.local_rank == 0:
                    print(line)
            self.log_info = print_line
        else:
            self.log_info = self.logger.log_info

    def weight_fix(self, way, refer_dict=None):

        # fix weights
        if way == 'checkpoint':
            assert refer_dict is not None
            for n, p in self.model.named_parameters():
                if n in refer_dict.keys():
                    p.requires_grad = False
        elif way == 'encoder':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n or 'enet' in n or 'fusion' in n:
                    p.requires_grad = False
        elif way == 'event':
            for n, p in self.model.named_parameters():
                if 'enet' in n or 'fusion' in n:
                    p.requires_grad = False
        elif way == 'eventencoder':
            for n, p in self.model.named_parameters():
                if 'enet' in n:
                    p.requires_grad = False
        elif way == 'eventfusion':
            for n, p in self.model.named_parameters():
                if 'fusion' in n:
                    p.requires_grad = False
        elif way == 'imageencoder':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n:
                    p.requires_grad = False
        elif way == 'raft':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n or 'update_block' in n:
                    p.requires_grad = False
        elif way == 'allencoder':
            for n, p in self.model.named_parameters():
                if 'enet' in n or 'fusion' in n or 'fnet' in n or 'cnet' in n:
                    p.requires_grad = False
        elif way == 'update':
            for n, p in self.model.named_parameters():
                if 'update_block' in n:
                    p.requires_grad = False

        self.log_info("Weight fix way: {} complete.".format(way if way != "" else "None"), "trainer")

    def partial_load(self, path, weight_fix=None, not_load=False):
        # partial parameters loading
        assert path != ''
        load_dict = torch.load(path, map_location=torch.device("cpu"))
        try:
            if "model" not in load_dict.keys():
                pretrained_dict = {k: v for k, v in load_dict.items() if k in self.model.state_dict().keys() \
                    and k != 'module.update_block.encoder.conv.weight' \
                    and k != 'module.update_block.encoder.conv.bias' \
                    and not k.startswith('module.update_block.flow_enc')}
            else:
                pretrained_dict = {k: v for k, v in load_dict.pop("model").items() if k in self.model.state_dict().keys() \
                    and k != 'module.update_block.encoder.conv.weight' \
                    and k != 'module.update_block.encoder.conv.bias' \
                    and not k.startswith('module.update_block.flow_enc')}
            assert len(pretrained_dict.keys()) > 0
            if not not_load:
                self.model.load_state_dict(pretrained_dict, strict=False)
                self.log_info("Partial load model from {} complete.".format(path), "trainer")
            else:
                self.log_info("Partial load dict from {} only for weight fix, but not load to model.".format(path), "trainer")
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check partial checkpoint path {}".format(path))

        self.weight_fix(weight_fix, pretrained_dict)

    def load(self, path, only_model=True):
        assert path != ''
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        try:
            if "model" not in state_dict.keys():
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict.pop("model"))
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(path))

        index = 0
        if not only_model:
            try:
                self.optimizer.load_state_dict(state_dict.pop("optimizer"))
            except:
                self.log_info("'optimizer' not in state_dict.keys(), skip it.", "trainer")

            try:
                self.lr_scheduler.load_state_dict(state_dict.pop("lr_scheduler"))
            except:
                self.log_info("'lr_scheduler' not in state_dict.keys(), skip it.", "trainer")

            try:
                index = state_dict.pop("index")
            except:
                self.log_info("'index' not in state_dict.keys(), set to 0.", "trainer")

            self.log_info("Load model/optimizer/index from {} complete, index {}".format(path, index), "trainer")
        else:
            self.log_info("Load model from {} complete, index {}".format(path, index), "trainer")

        return index

    def store(self, path, name, index=None):
        if path != "" and name != "":
            checkpoint = {}
            checkpoint["model"] = self.model.state_dict()
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            checkpoint["index"] = index

            ensure_folder(path)
            save_path = os.path.join(path, "{}_{}.pth".format(name, checkpoint["index"]))
            torch.save(checkpoint, save_path)
            self.log_info("<<< Save model to {} complete".format(save_path), "trainer")

    def run_steps(self, dataloader, dataloader_iterator, step=0, step_num=200):
        self.model.train()

        if self.args.local_rank == 0:
            self.bar = tqdm(total=step_num, position=0, leave=True)

        for _ in range(step, step + step_num):
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                batch = next(dataloader_iterator)

            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].cuda()

            # output = self.model(batch['img1'], batch['img2'], self.args.iters)
            output = self.model(batch, self.args.iters)
            loss = self.loss(output, batch)

            torch.distributed.barrier()
            reduced_loss = reduce_list(loss, self.args.nprocs)

            self.optimizer.zero_grad()

            self.scaler.scale(loss['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.scaler.update()

            if self.args.local_rank == 0:
                self.bar_update(reduced_loss)

            if self.logger is not None:
                self.logger.push(reduced_loss, 'loss', last=False)
                self.logger.push({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})

        if self.args.local_rank == 0:
            self.bar.close()
        return dataloader

    def run_epoch(self, dataloader):
        self.model.train()

        if self.args.local_rank == 0:
            self.bar = tqdm(total=len(dataloader), position=0, leave=True)
        for index, batch in enumerate(dataloader):

            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].cuda(self.args.gpus[self.args.local_rank] \
                        if self.args.local_rank != -1 else 0, non_blocking=True)

            # output = self.model(batch['img1'], batch['img2'], self.args.iters)
            output = self.model(batch, self.args.iters)
            loss = self.loss(output, batch)

            self.optimizer.zero_grad()

            torch.distributed.barrier()
            reduced_loss = reduce_list(loss, self.args.nprocs)

            self.scaler.scale(loss['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.scaler.update()

            if self.args.local_rank == 0:
                self.bar_update(reduced_loss)

            if self.logger is not None:
                self.logger.push(reduced_loss, 'loss', last=False)
                self.logger.push({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})

        if self.args.local_rank == 0:
            self.bar.close()

    def bar_update(self, loss):
        loss_description = ""
        for data, key in zip(loss.values(), loss.keys()):
            loss_description += "{}:{:5.4f}, ".format(key, data) if 'px' not in key else "{}:{:4.3f}, ".format(key, data)
        self.bar.set_description(loss_description)
        self.bar.update(1)
