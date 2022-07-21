import sys
import os
import os.path as osp
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from config import config
from network.deeplab import DeeplabV2
from criterion import CriterionOhemCE, CriterionWeightedCE
from optimizer import Optimizer
from evaluate import Evaluator
from dataloader import get_gtav_train_loader, get_city_train_loader

cudnn.benchmark = True

class Trainer(object):
    def __init__(self):
        # Log setting
        self.save_root = config.save_root
        self.time = config.time
        self.logger, self.writer = self.init_log_stream()

        # DataLoader setting
        self.source_loader = get_gtav_train_loader()
        self.target_loader = get_city_train_loader()
        self.source_loader_iter = iter(self.source_loader)
        self.target_loader_iter = iter(self.target_loader)

        # Criterion setting
        if config.source_criterion == 'ce':
            self.source_criterion = nn.CrossEntropyLoss(weight=None, ignore_index=config.ignore_label)
        elif config.source_criterion == 'ohemce':
            min_kept = int(config.batch_size * config.crop_size[0] * config.crop_size[1] // 16)
            self.source_criterion = CriterionOhemCE(thresh=0.7, min_kept=min_kept, ignore_index=config.ignore_label)
        elif config.source_criterion == 'weightedce':
            weight = torch.load(config.source_weight_path).cuda()
            self.source_criterion = CriterionWeightedCE(weight=weight, ignore_index=config.ignore_label)

        # Model setting
        self.num_classes = config.num_classes
        self.adabn = config.adabn
        self.model = DeeplabV2(num_classes=self.num_classes)
        self.model.cuda()
        self.model.train()

        # Optimizer setting
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        self.iters_per_epoch = config.iters_per_epoch
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.stop_iters = config.num_epochs * config.iters_per_epoch
        self.optimizer = Optimizer(
            params=self.model.optim_parameters(config.lr, config.weight_decay),
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
        self.last_path = osp.join(self.ckpt_path, 'epoch-last.pth')

    def start(self):
        # Print information
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
            if (self.current_epoch % 1 == 0):
                save_path = osp.join(self.ckpt_path, 'epoch-{}.pth'.format(self.current_epoch))
                torch.save(self.model.state_dict(), save_path)
                self.logger.warning('Saving checkpoint to {}'.format(save_path))
                if osp.isdir(self.last_path) or osp.isfile(self.last_path) or osp.islink(self.last_path):
                    os.remove(self.last_path)
                os.system('ln -s {} {}'.format(save_path, self.last_path))

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
            ######################################
            #           Source Domain            #
            ######################################
            source_img, source_gt = next(self.source_loader_iter)
            source_img = source_img.cuda()
            source_gt = source_gt.cuda()
            input_size = source_img.size()[2:]

            # Source Forward
            self.optimizer.zero_grad()
            source_out = self.model(source_img)
            source_out = F.interpolate(source_out, size=input_size, mode='bilinear', align_corners=True)
            source_loss = self.source_criterion(source_out, source_gt)

            # SourceBackward
            source_loss.backward()

            ######################################
            #           Target Domain            #
            ######################################
            if self.adabn:
                with torch.no_grad():
                    target_img, _ = next(self.target_loader_iter)
                    target_img = target_img.cuda()

                    # AdaBN
                    self.model(target_img)

            # Optimize all parameter
            self.optimizer.step()

            # Print information and save log file
            self.current_iter += 1
            print_str = 'Epoch-{}/{}'.format(self.current_epoch, self.num_epochs).ljust(12) \
                        + 'Iter-{}/{}'.format(self.current_iter, self.stop_iters).ljust(12) \
                        + 'lr=%.2e ' % self.optimizer.lr \
                        + 'loss=%.4f' % source_loss.item()
            pbar.set_description(print_str, refresh=False)
            if self.current_iter % config.log_iter == 0:
                self.logger.info(print_str)
                self.writer.add_scalar('train/learning_rate', self.optimizer.lr, self.current_iter)
                self.writer.add_scalar('train/source_loss', source_loss.item(), self.current_iter)

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
