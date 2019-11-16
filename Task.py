

import numpy as np

from torch.utils.data import DataLoader
import MiniDataset as md
import torch
import os


def get_random_class_sample_order(n_way, k_shot):
    x = []
    for i in range(n_way):
        x.append(i)
    return np.random.permutation(np.array(k_shot * x))


class MiniTask():

    def __init__(self, n_way, k_shot, batch_size, root_dir, from_what_datasets, priority=1):

        classes = np.arange(n_way)

        permutation = np.random.permutation(n_way)

        self.classes = classes[permutation]

        self.n_way = n_way

        self.k_shot = k_shot

        self.num_of_batches = self.n_way * self.k_shot / batch_size

        self.batch_size = batch_size

        self.root_dir = root_dir

        self.datasets = []

        self.from_what_datasets = from_what_datasets

        self.priority = priority

        if len(from_what_datasets) is not n_way:
            raise Exception('Length of from_what_datasets is not n_way')


        if (n_way*k_shot)%batch_size is not 0:
            message = 'Batch size ' + str(batch_size) +' does not divide ' + str(n_way) + '*' + str(k_shot)
            raise Exception(message)


        for idx in from_what_datasets:

            d = md.MiniDataset(root_dir + str(idx), trans=md.ToTensor())
            self.datasets.append(d)

        self.dataset_sample_order = get_random_class_sample_order(
            n_way=n_way, k_shot=k_shot)

        self.classes_shuffled = np.random.permutation(np.arange(n_way))

    def sample_batch(self):
        batch = []
        labels = []
        for i in range(self.batch_size):

            batch.append(
                self.datasets[self.dataset_sample_order[i]].__getrandomitem__())
            labels.append(self.classes_shuffled[self.dataset_sample_order[i]])
        self.dataset_sample_order = self.dataset_sample_order[self.batch_size:]

        if self.dataset_sample_order.size == 0:
            self.dataset_sample_order = get_random_class_sample_order(
                n_way=self.n_way, k_shot=self.k_shot)

        labels = torch.Tensor(labels)
        batch = torch.stack(batch)

        return batch.type('torch.FloatTensor').cuda(), labels.long()


        #return batch, labels
    def sample_heldoutbatch(self):
        batch = []
        labels = []
        for i in range(self.batch_size):

            batch.append(
                self.datasets[self.dataset_sample_order[i]].__getrandomheldoutitem__())
            labels.append(self.classes_shuffled[self.dataset_sample_order[i]])
        self.dataset_sample_order = self.dataset_sample_order[self.batch_size:]

        if self.dataset_sample_order.size == 0:
            self.dataset_sample_order = get_random_class_sample_order(
                n_way=self.n_way, k_shot=self.k_shot)

        labels = torch.Tensor(labels)
        batch = torch.stack(batch)

        return batch.type('torch.FloatTensor').cuda(), labels.long()


    def sample_heldoutbatches(self, k_shot_multiple=1):
        batches = []

        for i in range(int((self.num_of_batches)*k_shot_multiple)):
            batches.append(self.sample_heldoutbatch())

        return batches

    def sample_batches(self, k_shot_multiple=1):
        batches = []

        for i in range(int((self.num_of_batches)*k_shot_multiple)):
            batches.append(self.sample_batch())

        return batches


class MiniTaskDistribution():

    def __init__(self, n_way, k_shot, tasks_batch_size, phase):
        self.root_dir = '~/MAML/maml-s/imagenet/processed_images/'
        self.datasets_dir = self.root_dir + phase + "/"
        self.n_way = n_way

        self.k_shot = k_shot
        self.tasks_batch_size = tasks_batch_size
        self.phase = phase
        self.total_num_of_datasets = len(os.listdir(self.datasets_dir)) - 1



        if n_way > self.total_num_of_datasets:
            raise Exception("N way is greater than total number of classes")

        self.random_dataset_order = np.random.permutation(
            np.arange(self.total_num_of_datasets))


    def sample_tasks(self, num_of_tasks):

        data_sets_for_tasks = []

        for i in range(num_of_tasks):

            if self.n_way > len(self.random_dataset_order):
                new_dataset_order = np.random.permutation(
                    np.arange(self.total_num_of_datasets))
                new_dataset_order = np.delete(
                    new_dataset_order, self.random_dataset_order)
                self.random_dataset_order = np.append(
                    self.random_dataset_order, new_dataset_order)
            data_sets_for_tasks.append(self.random_dataset_order[0:self.n_way])
            self.random_dataset_order = self.random_dataset_order[self.n_way:]

        tasks = []

        for i in range(num_of_tasks):
            tasks.append(MiniTask(from_what_datasets=data_sets_for_tasks[i],
                                  n_way=self.n_way,
                                  k_shot=self.k_shot,
                                  batch_size=self.tasks_batch_size,
                                  root_dir=self.datasets_dir))
        return np.array(tasks)
