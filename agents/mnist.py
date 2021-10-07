"""
Mnist Main agent, as mentioned in the tutorial
"""
import os
import copy
from copy import deepcopy
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from agents.base import BaseAgent

from graphs.models.mnist import Mnist
from datasets.mnist import MnistDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class MnistAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = Mnist()
        self.best_model_state = None

        # define data_loader
        self.data_loader = MnistDataLoader(config=config)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        self.best_optimizer_state = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_epoch = 0
        self.best_metric = 0

        # initialize loss
        self.train_loss = 0
        self.test_loss = 0
        self.best_loss = 1e2

        # early stopping
        self.patience = 2
        self.trigger_times = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.checkpoint_file = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_file)
        self.load_checkpoint(self.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(self.config.log_dir)

        # Add Graph(Model)
        data, target = next(iter(self.data_loader.train_loader))
        data = data.to(self.device)
        grid = torchvision.utils.make_grid(data)
        self.summary_writer.add_image('images', grid, 0)
        self.summary_writer.add_graph(self.model, data)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except FileNotFoundError:
            print('Checkpoint not found!\nStart from scratch\n')

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """ 
        if is_best:
            torch.save({
                'epoch':self.best_epoch,
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.best_optimizer_state,
                'loss': self.best_loss,
                }, file_name)

    def early_stopping(self):
        """
        Early Stopping
        source: https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
        return: boolian
        """
        if self.test_loss >= self.best_loss:
            self.trigger_times += 1

            if self.trigger_times >= self.patience:
                print('Early stopping!\nStart to test process.')
                return True
        else:
            self.trigger_times = 0
            self.best_loss = self.test_loss
            return False

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        is_best = False
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()
            
            if self.test_loss < self.best_loss:
                is_best = True
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.best_loss = self.test_loss
                self.best_epoch = self.current_epoch
                
            # Save the current best checkpoint every log_interval epochs
            # Write Info to TensorBoard every log_interval epochs
            if self.current_epoch % self.config.log_interval == 0:
                self.save_checkpoint(self.checkpoint_file, is_best)
                self.summary_writer.flush()
                is_best = False

            if self.early_stopping():
                break

            self.current_epoch += 1
        
        # Save the best checkpoint every log_interval epochs
        self.save_checkpoint(self.checkpoint_file, is_best)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        self.train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
            self.train_loss += loss.item()
        
        self.train_loss /= len(self.data_loader.train_loader.dataset)
        self.logger.info('\nTrain set: Average loss: {:.4f}\n'.format(self.train_loss))
        self.summary_writer.add_scalar('Loss/train', self.train_loss, self.current_epoch+1)

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
        self.summary_writer.add_scalar('Loss/test', self.test_loss, self.current_epoch+1)

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # log hparams in tensorboard
        # hparams = ['batch_size', 'test_batch_size', 'learning_rate']

        # self.summary_writer.add_hparams({k:v for k,v in self.config.items() if k in hparams},
        #                                 {'hparam/accuracy': self.train_loss, 'hparam/loss': self.test_loss})
        self.summary_writer.close()