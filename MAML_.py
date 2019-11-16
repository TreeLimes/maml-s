import torch
import torch.nn as nn
from collections import OrderedDict
import Task as T

import torch.nn.functional as F
import numpy as np
import copy
from utils import init_params


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def permute_tasks(tasks, priority):
    perm = np.random.permutation(np.arange(len(tasks)))

    return tasks[perm], priority[perm]


class MAMLModel(nn.Module):
    def __init__(self, nway=5, base_lr=1e-2):
        super(MAMLModel, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2,2,0)

        self.conv2 = nn.Conv2d(32,32,3,1,0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2,2,0)

        self.conv3 = nn.Conv2d(32,32,3,1,0)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(2,2,0)

        self.conv4 = nn.Conv2d(32,32,3,1,0)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        self.maxpool4 = nn.MaxPool2d(2,1,0)

        self.fc = nn.Linear(800, 5)

        self.num_of_layers = 17

        init_params(self)

        self.base_lr = base_lr


    def forward(self, x, weights):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.maxpool3(x)
        x = self.bn4(self.relu4(self.conv4(x)))
        x = self.maxpool4(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x
    def parameterised(self, x, weights, bn_training=True):
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                x = F.conv2d(x, weights[idx], weights[idx+1], m.stride, m.padding)
                idx += 2
            elif isinstance(m, nn.BatchNorm2d):
                x = F.batch_norm(x, m.running_mean, m.running_var, weights[idx], weights[idx+1], bn_training)
                idx += 2
            elif isinstance(m, nn.Linear):
                x = x.view(x.shape[0], -1)
                x = F.linear(x, weights[idx], weights[idx+1])
                idx += 2
            elif isinstance(m, nn.ReLU):
                x = F.relu(x)
            elif isinstance(m, nn.MaxPool2d):
                x = F.max_pool2d(x, m.kernel_size, m.stride, m.padding)
        return x
    def parameterised_excluding_lastlayer(self, x, weights, bn_training=True):
        idx = 0
        for i, m in enumerate(self.modules()):
            if i is not self.num_of_layers:
                if isinstance(m, nn.Conv2d):
                    x = F.conv2d(x, weights[idx], weights[idx+1], m.stride, m.padding)
                    idx += 2
                elif isinstance(m, nn.BatchNorm2d):
                    x = F.batch_norm(x, m.running_mean, m.running_var, weights[idx], weights[idx+1], bn_training)
                    idx += 2
                elif isinstance(m, nn.Linear):
                    x = x.view(x.shape[0], -1)
                    x = F.linear(x, weights[idx], weights[idx+1])
                    idx += 2
                elif isinstance(m, nn.ReLU):
                    x = F.relu(x)
                elif isinstance(m, nn.MaxPool2d):
                    x = F.max_pool2d(x, m.kernel_size, m.stride, m.padding)
        return x




class MAML():
    def __init__(self, model, inner_lr, meta_lr, num_iterations, inner_steps=1, tasks_per_meta_batch=4, num_of_epochs=1500, train_tasks=None, val_tasks=None, priority=None):

        # important objects
        self.model = model.cuda()
        # the maml weights we will be meta-optimising
        self.weights = list(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.meta_optimiser = torch.optim.Adam(
            self.weights, meta_lr, [0.9, 0.99])
        self.num_of_val_tasks = 10
        self.num_of_test_tasks = 10
        self.k_shot_factor_for_qry = 1

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

        # with the current design of MAML, >1 is unlikely to work well
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.num_of_epochs = num_of_epochs
        # get task not going to be seen while training
        self.distribution_train_tasks = T.MiniTaskDistribution(
            n_way=5, k_shot=5, tasks_batch_size=25, phase="train")
        self.distribution_val_tasks = T.MiniTaskDistribution(
            n_way=5, k_shot=5, tasks_batch_size=25, phase="val")
        self.distribution_test_tasks = T.MiniTaskDistribution(
            n_way=5, k_shot=5, tasks_batch_size=25, phase="test")

        self.num_iterations = num_iterations

        if train_tasks is None and val_tasks is None:
            self.train_tasks = self.distribution_train_tasks.sample_tasks(num_iterations*self.tasks_per_meta_batch)
            self.val_tasks = self.distribution_val_tasks.sample_tasks(self.num_of_val_tasks)
        else:
            self.train_tasks = train_tasks
            self.val_tasks = val_tasks

        if priority is None:
            self.priority = np.ones(len(self.train_tasks))
        else:
            self.priority = priority

        self.test_tasks = self.distribution_test_tasks.sample_tasks(self.num_of_test_tasks)

        self.avg_train_losses = []
        self.avg_train_accs = []
        self.avg_val_losses = []
        self.avg_val_accs = []


    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        #inner_optimizer = torch.optim.Adam(
        #temp_weights, self.meta_lr, [0.9, 0.99])

        # perform training on data sampled from task

        for step in range(self.inner_steps):
            # how do we use adam in the inner update?
            for x_batch, y_batch in task.sample_batches():
                out = self.model.parameterised(
                    x_batch, temp_weights)

                loss = self.criterion(out, y_batch)

                # compute grad and update inner loop weights
                grad = torch.autograd.grad(loss, temp_weights)
                temp_weights = [w - self.inner_lr *
                                g for w, g in zip(temp_weights, grad)]

        # find meta loss on batch of meta data, maybe should approach data one at a time

        loss = 0
        acc = 0

        model_labels = []
        labels = []

        for x_batch, y_batch in task.sample_batches(self.k_shot_factor_for_qry):

            out = self.model.parameterised(
                x_batch, temp_weights)

            loss += self.criterion(out, y_batch)
            for out_ in out:
                model_labels.append(torch.max(out_, 0)[1].item())
            for label in y_batch:
                labels.append(label.item())
        num_of_correct = np.count_nonzero(
                (np.array(model_labels) - np.array(labels)) == 0)

        acc = num_of_correct / len(labels)

        return loss / (task.num_of_batches*self.k_shot_factor_for_qry), acc

    def train(self):

        # Epoch
        for epoch in range(self.num_of_epochs):

            train_loss_avg = 0
            train_acc_avg = 0
            #keep prioty with the right task
            all_tasks, self.priority = permute_tasks(self.train_tasks, self.priority)

            for iteration in range(1, self.num_iterations + 1):
                self.meta_optimiser.zero_grad()
                train_loss = 0
                train_acc = 0

                tasks = all_tasks[(iteration-1)*self.tasks_per_meta_batch:(iteration-1)*self.tasks_per_meta_batch + self.tasks_per_meta_batch]
                priorities = self.priority[(iteration-1)*self.tasks_per_meta_batch:(iteration-1)*self.tasks_per_meta_batch + self.tasks_per_meta_batch]
                # compute meta loss
                for count, task in enumerate(tasks):
                    # new tasks every iteration
                    loss, acc = self.inner_loop(task)
                    loss *= priorities[count]

                    train_loss += loss
                    train_acc += acc

                # compute meta gradient of loss with respect to maml weights

                train_loss /= self.tasks_per_meta_batch
                train_acc /= self.tasks_per_meta_batch
                meta_grads = torch.autograd.grad(train_loss, self.weights)

                # assign meta gradient to weights and take optimisation step
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g

                self.meta_optimiser.step()

                train_loss_avg += train_loss.item()
                train_acc_avg += train_acc
            train_loss_avg /= iteration
            train_acc_avg /= iteration

            val_loss = 0
            val_acc = 0

            # dangerous since torch.no_grad() not called
            for task in self.val_tasks:

                loss, acc = self.inner_loop(task)

                val_loss += loss.item()
                val_acc += acc

            self.avg_train_losses.append(train_loss_avg)
            self.avg_train_accs.append(train_acc_avg)
            self.avg_val_losses.append(val_loss / len(self.val_tasks))
            self.avg_val_accs.append(val_acc / len(self.val_tasks))

