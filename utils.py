import numpy as np
import torch
import torch.nn as nn
import MAML_ as M
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))


def fine_tune(task, maml, base_model_weights):

    temp_weights = [w.clone() for w in base_model_weights]

    for step in range(maml.inner_steps):
        # how do we use adam in the inner update?
        for x_batch, y_batch in task.sample_heldoutbatches():
            out = maml.model.parameterised(
                x_batch, temp_weights)

            loss = maml.criterion(out, y_batch)

            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - maml.inner_lr *
                            g for w, g in zip(temp_weights, grad)]
            maml.meta_optimiser.zero_grad()
    return temp_weights


def shared_embed(t1, t2, maml, base_model_weights):

    finetuned1 = fine_tune(t1, maml, base_model_weights)
    finetuned2 = fine_tune(t2, maml, base_model_weights)

    o11, o21 = concatenated_outputs(finetuned1, finetuned2, t1.sample_heldoutbatches(), maml)
    o12, o22 = concatenated_outputs(finetuned1, finetuned2, t2.sample_heldoutbatches(), maml)

    o1 = np.concatenate([o11, o12], axis=None)
    o2 = np.concatenate([o21, o22], axis=None)

    return o1, o2


def concatenated_outputs(w1, w2, data, maml):
    outputs1 = []
    outputs2 = []

    for x_batch, y_batch in data:
        o1 = maml.model.parameterised_excluding_lastlayer(
                x_batch, w1)
        o2 = maml.model.parameterised_excluding_lastlayer(
                x_batch, w2)
        o1 = o1.view(o1.size(0), -1)
        o2 = o2.view(o2.size(0), -1)
        outputs1.append(o1.detach())
        outputs2.append(o2.detach())

    output1 = np.concatenate(outputs1, axis=None)
    output2 = np.concatenate(outputs2, axis=None)

    return output1, output2


def get_closest_task_index(all_distances, already_used_tasks):
    min_index = 0
    min_distance = -1
    distances = np.zeros(len(all_distances[0]))
    for distance in all_distances:
        distances += distance
    distances /= len(all_distances)

    for i, d in enumerate(distances):
        if i not in already_used_tasks:
            if d < min_distance or min_distance == -1:
                min_distance = d
                min_index = i

    return min_index

def get_trajectories(n, maml):
    base_model = maml.model

    tasks = maml.train_tasks

    base_model_weights = list(base_model.parameters())

    all_distances = []

    all_reduced_embeddings = []

    all_embeddings = []

    used_root_indices = []

    closest_task_index = 0

    for i in range(n):
        root_task = None
        root_indx = 0
        if i == 0:
            root_indx = np.random.randint(low=0, high=len(tasks))
            root_task = tasks[root_indx]
        else:
            root_indx = closest_task_index
            root_task = tasks[root_indx]

        embeddings = []
        for j in range(len(tasks)):
            o1, o2 = shared_embed(root_task, tasks[j], maml, base_model_weights)
            #root_embedding
            embeddings.append(o1)
            #leaf_embedding
            embeddings.append(o2)

        all_embeddings.append(embeddings)

        reduced_embeddings = TSNE(n_components=2).fit_transform(embeddings)

        all_reduced_embeddings.append(reduced_embeddings)

        used_root_indices.append(root_indx)
        distances = []

        #compare root emebedding to leaf
        for i in range(len(tasks)):
            distances.append(dist(reduced_embeddings[i*2], reduced_embeddings[i*2 + 1]))

        #normalize distances
        distances = 2*(np.ones(len(distances)) - distances / np.max(distances))

        all_distances.append(distances)

        closest_task_index = get_closest_task_index(all_distances, used_root_indices)
    return used_root_indices, all_distances, all_reduced_embeddings, all_embeddings

def graph_data(used_root_indices, data, all_distances):
    fig, axs = plt.subplots(len(data))

    for i, d in enumerate(data):
        axs[i].scatter(d[:,0], d[:,1])
    fig.subplots_adjust(hspace=2)
    plt.show()

def distance_between_tasks(tasks, used_root_indices, t, maml):
    distances = []
    base_model_weights = list(maml.model.parameters())
    embeddings = []
    for j in range(len(tasks)):
        o1, o2 = shared_embed(t, tasks[j], maml, base_model_weights)
        #embeded t
        embeddings.append(o1)
        #leaf_embedding
        embeddings.append(o2)
    reduced_embeddings = TSNE(n_components=2).fit_transform(embeddings)

    for i in used_root_indices:
        distances.append(dist(reduced_embeddings[i*2], reduced_embeddings[i*2 + 1]))

    distances = 2*(np.ones(len(distances)) - distances / np.max(distances))

    return distances



def num_param(model):
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    return num


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

