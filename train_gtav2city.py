import sys
import os
import os.path as osp
import logging
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from tensorboardX import SummaryWriter
import numpy as np

from config import config
from network.deeplab import DeeplabV2
from criterion import CriterionOhemCE, CriterionWeightedCE, CriterionCEKD, CriterionWeightedCEKD,\
    CriterionWeightNeutralKD,CriterionWeightFuse, CriterionLabeledKD
from optimizer import Optimizer
from evaluate import Evaluator
from dataloader import get_gtav_train_loader, get_city_train_loader

from datasets.augmentations import RandomCutMix, RandomAffine

cudnn.benchmark = True

class Trainer(object):
    def __init__(self):
        # Log setting
        self.save_root = config.save_root
        self.time = config.time
        self.log_iter = config.log_iter
        self.logger, self.writer = self.init_log_stream()

        # Distributed training setting
        self.num_gpus = torch.cuda.device_count()
        args = self.parse_args()
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=torch.cuda.device_count(),
            rank=args.local_rank)

        # DataLoader setting
        self.source_loader = get_gtav_train_loader()
        self.target_loader = get_city_train_loader()
        self.source_loader_iter = iter(self.source_loader)
        self.target_loader_iter = iter(self.target_loader)
        self.batch_size = config.batch_size

        # Criterion setting
        if config.source_criterion == 'ce':
            self.source_criterion_out = nn.CrossEntropyLoss(weight=None, ignore_index=config.ignore_label)
            self.source_criterion_aux = nn.CrossEntropyLoss(weight=None, ignore_index=config.ignore_label)
        elif config.source_criterion == 'ohemce':
            min_kept = int(config.batch_size * config.crop_size[0] * config.crop_size[1] // 16)
            self.source_criterion_out = CriterionOhemCE(thresh=0.7, min_kept=min_kept, ignore_index=config.ignore_label)
            self.source_criterion_aux = CriterionOhemCE(thresh=0.7, min_kept=min_kept, ignore_index=config.ignore_label)
        elif config.source_criterion == 'weightedce':
            weight = torch.load(config.source_weight_path).cuda()
            self.source_criterion = CriterionWeightedCE(weight=weight, ignore_index=config.ignore_label)
        elif config.source_criterion == 'labeledkd':
            weight = torch.load(config.source_weight_path).cuda()
            self.source_criterion = CriterionLabeledKD(alpha=config.alpha, weight=weight, ignore_index=config.ignore_label)

        if config.target_criterion == 'ce':
            self.target_criterion_out = nn.CrossEntropyLoss(weight=None, ignore_index=config.ignore_label)
        elif config.target_criterion == 'ohemce':
            min_kept = int(config.batch_size * config.crop_size[0] * config.crop_size[1] // 16)
            self.target_criterion_out = CriterionOhemCE(thresh=0.7, min_kept=min_kept, ignore_index=config.ignore_label)
        elif config.target_criterion == 'weightedce':
            weight = torch.load(config.target_weight_path).cuda()
            self.target_criterion = CriterionWeightedCE(weight=weight, ignore_index=config.ignore_label)
        elif config.target_criterion == 'cekd':
            self.target_criterion_out = CriterionCEKD()
        elif config.target_criterion == 'weightedcekd':
            weight = torch.load(config.target_weight_path).cuda()
            self.target_criterion_out = CriterionWeightedCEKD(weight=weight)
        elif config.target_criterion == 'wnce':
            weight = torch.load(config.target_weight_path).cuda()
            self.target_criterion = CriterionWeightNeutralKD(weight=weight)

        # Model setting
        self.num_classes = config.num_classes
        self.model = DeeplabV2(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(config.ckpt_path))
        self.model.cuda()
        self.model.train()
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[args.local_rank, ],
            output_device=args.local_rank
        )

        self.teacher = DeeplabV2(num_classes=self.num_classes)
        self.teacher.load_state_dict(torch.load(config.ckpt_path))
        self.teacher.cuda()
        self.teacher.eval()
        self.teacher = nn.parallel.DistributedDataParallel(
            self.teacher,
            device_ids=[args.local_rank, ],
            output_device=args.local_rank
        )

        self.net_momentum_param = config.net_momentum_param
        self.net_momentum_iters = config.net_momentum_iters

        self.cutmix = config.cutmix
        self.affine = config.affine
        self.aug_cutmix = RandomCutMix()
        self.aug_affine = RandomAffine()

        # Optimizer setting
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        self.iters_per_epoch = config.iters_per_epoch
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.stop_iters = config.num_epochs * config.iters_per_epoch
        self.optimizer = Optimizer(
            params=self.model.module.optim_parameters(config.lr, config.weight_decay),
            base_lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            current_iter=self.current_iter,
            total_iter=self.num_iters,
            lr_power=config.lr_power,
            policy=config.lr_policy
        )

        # Evaluate agent
        self.agent_eval = Evaluator(logger=self.logger, writer=self.writer)

        # Checkpoint and resume setting
        self.best_mIoU = 0
        self.best_epoch = 0
        self.ckpt_path = osp.join(self.save_root, 'ckpt')
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def start(self):
        # Print information
        if dist.get_rank() == 0:
            self.logger.warning('Global configuration as follows:')
            for key, val in config.items():
                self.logger.warning("{:16} {}".format(key, val))

        # Start to training
        self.train()

        # Close writer
        self.writer.close()

    def train(self):
        # Epochs during training
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            # Save checkpoint
            if dist.get_rank() == 0:
                save_path = osp.join(self.ckpt_path, 'epoch-{}.pth'.format(self.current_epoch))
                state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(state, save_path)

                self.logger.warning('Saving checkpoint to {}'.format(save_path))

                # Evaluate one epoch
                mIoU = self.agent_eval.start(save_path)
                if mIoU > self.best_mIoU:
                    self.best_mIoU = mIoU
                    self.best_epoch = self.current_epoch
                    self.logger.warning('New best mIoU {} is at epoch-{}'.format(self.best_mIoU, self.best_epoch))
                else:
                    self.logger.warning('The best mIoU {} is still at epoch-{}'.format(self.best_mIoU, self.best_epoch))
                self.writer.add_scalar('train/city_mIoU', mIoU, self.current_epoch)

    def train_one_epoch(self):
        pbar = tqdm(range(self.iters_per_epoch), file=sys.stdout)
        for idx in pbar:
            self.optimizer.zero_grad()
            # Total Forward
            source_img, source_lbl, source_ori = next(self.source_loader_iter)
            target_img, _, target_ori = next(self.target_loader_iter)
            source_img, source_ori = source_img.cuda(), source_ori.cuda()
            target_img, target_ori = target_img.cuda(), target_ori.cuda()
            source_lbl = source_lbl.cuda()
            input_size = source_lbl.size()[1:]
            with torch.no_grad():
                ori = torch.cat((source_ori, target_ori))
                soft = self.teacher(ori)
                soft = F.interpolate(soft, size=input_size, mode='bilinear', align_corners=True)
                source_soft, target_soft = soft[:self.batch_size], soft[self.batch_size:]
                _, target_lbl = torch.max(target_soft, dim=1)

                if self.cutmix:
                    source_img, source_lbl, source_soft, target_img, target_lbl, target_soft = self.aug_cutmix(
                        source_img, source_lbl, source_soft, target_img, target_lbl, target_soft
                    )

                target_mask = torch.ones_like(target_lbl).float()
                if self.affine:
                    target_img, target_lbl, target_soft, target_mask = self.aug_affine(
                        target_img, target_lbl, target_soft
                    )

            #img = torch.cat((source_img, target_img))
            source_out = self.model(source_img)
            source_out = F.interpolate(source_out, size=input_size, mode='bilinear', align_corners=True)
            #source_out, target_out = out[:self.batch_size], out[self.batch_size:]
            target_out = self.model(target_img)
            target_out = F.interpolate(target_out, size=input_size, mode='bilinear', align_corners=True)

            # Source Loss
            source_loss = self.source_criterion(source_out, source_lbl)
            source_loss.backward()

            # Target Loss
            target_loss = self.target_criterion(target_out, target_soft, target_mask)
            target_loss.backward()

            # Total Loss
            #loss = source_loss + target_loss

            # Optimize
            #loss.backward()
            self.optimizer.step()

            # Print information and save log file
            self.current_iter += 1
            print_str = 'Epoch-{}/{}'.format(self.current_epoch, self.num_epochs).ljust(12) \
                        + 'Iter-{}/{}'.format(self.current_iter, self.num_iters).ljust(12) \
                        + 'lr=%.2e ' % self.optimizer.lr \
                        + 'sloss=%.4f' % source_loss.item() \
                        + 'tloss=%.4f' % target_loss.item()

            pbar.set_description(print_str, refresh=False)
            if self.current_iter % self.log_iter == 0 and dist.get_rank() == 0:
                self.logger.info(print_str)
                self.writer.add_scalar('train/learning_rate', self.optimizer.lr, self.current_iter)
                self.writer.add_scalar('train/source_loss', source_loss.item(), self.current_iter)
                self.writer.add_scalar('train/target_loss', target_loss.item(), self.current_iter)

            if self.current_iter % self.net_momentum_iters == 0:
                self.momentum_update()

    def momentum_update(self):
        """Momentum update"""
        teacher_dict = self.teacher.state_dict()
        student_dict = self.model.state_dict()
        for key, val in student_dict.items():

            if key.split(".")[-1] in ("weight", "bias", "running_mean", "running_var"):
                teacher_dict[key].mul_(self.net_momentum_param)
                teacher_dict[key].add_(val * (1. - self.net_momentum_param))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', default=2, type=int, help='node rank for distributed training')
        args = parser.parse_args()
        return args

    def reduce_tensor(self, tensor):
        reduce_tensor = tensor.clone()
        dist.all_reduce(reduce_tensor, dist.ReduceOp.SUM)
        reduce_tensor.div_(self.num_gpus)
        return reduce_tensor.item()

    def init_log_stream(self):
        self.save_root = osp.join(self.save_root, 'train', self.time)
        log_path = osp.join(self.save_root, 'log')
        if not osp.exists(log_path):
            os.makedirs(log_path)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_path, 'train_log.txt'))
        ch = logging.StreamHandler()
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        # Writer setting
        data_path = osp.join(self.save_root, 'board')
        writer = SummaryWriter(data_path)

        return logger, writer

if __name__ == '__main__':
    # Training
    agent_train = Trainer()
    agent_train.start()
