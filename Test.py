import MAML_ as M
import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


model = M.MAMLModel()

base_maml = M.MAML(model, inner_lr=0.01, meta_lr=0.001, num_iterations=100, inner_steps=5, tasks_per_meta_batch=4, num_of_epochs=150)

base_maml.train()

#must be after train so tasks aren't permuted
train_tasks = base_maml.train_tasks
val_tasks = base_maml.val_tasks
test_tasks = base_maml.test_tasks

print("Trained base model")

used_root_indices, all_distances, all_reduced_embeddings, all_embeddings = utils.get_trajectories(5, base_maml)
print(all_distances)
mamls = []

for priority in all_distances:
    model_copy = type(base_maml.model)()
    model_copy.load_state_dict(base_maml.model.state_dict())
    m = M.MAML(model_copy, inner_lr=0.01, meta_lr=0.001, num_iterations=100, inner_steps=5, tasks_per_meta_batch=4, num_of_epochs=150, train_tasks=train_tasks, val_tasks=val_tasks, priority=priority)
    m.train()
    mamls.append(m)
    print("Trained")

#test_loss = []
test_accs = []
s_test_accs = []
s_base_test_accs = []
#base_test_loss = []
base_test_accs = []
distances = []

for t in test_tasks:
    d = utils.distance_between_tasks(train_tasks, used_root_indices, t, base_maml)
    distances.append(d)
    closest_task_indx = np.argmin(d)

    base_fine_tuned_weights = utils.fine_tune(t, base_maml, base_maml.weights)

    fine_tuned_weights = []

    for m in mamls:
        fine_tuned_weights.append(utils.fine_tune(t, m, m.weights))


    accs = []
    base_accs = []

    for i, m in enumerate(mamls):

        base_model_labels = []
        #base_losses = []

        model_labels = []
        labels = []
        #losses = []

        for x_batch, y_batch in t.sample_batches(m.k_shot_factor_for_qry):

            out = m.model.parameterised(
                x_batch, fine_tuned_weights)
            base_out = base_maml.model.parameterised(
                x_batch, base_fine_tuned_weights)

            #losses.append(m.criterion(out, y_batch).item())
            #base_losses.append(base_maml.criterion(base_out, y_batch).item())

            for out_ in out:
                model_labels.append(torch.max(out_, 0)[1].item())
            for label in y_batch:
                labels.append(label.item())

            for out_ in base_out:
                base_model_labels.append(torch.max(out_, 0)[1].item())
            m.meta_optimiser.zero_grad()
        #test_loss.append(np.mean(losses))

        #base_test_loss.append(np.mean(base_losses))

        num_of_correct = np.count_nonzero(
                (np.array(model_labels) - np.array(labels)) == 0)
        base_num_of_correct = np.count_nonzero(
                (np.array(base_model_labels) - np.array(labels)) == 0)

        acc = num_of_correct / len(labels)
        base_acc = base_num_of_correct / len(labels)

        accs.append(acc)
        base_accs.append(base_acc)
        if i == closest_task_indx:
            s_test_accs.append(accs)
            s_base_test_accs.append(base_acc)


    test_accs.append(accs)
    base_test_accs.append(base_accs)


for i, dist, accs, base_accs in enumerate(zip(distances, test_accs, base_test_accs)):
    if not os.path.exists("graphs/task" + str(i)):
        os.mkdir("graphs/task"+str(i))
    plt.plot(dist, accs, 'bo', label="S accs")
    plt.plot(dist, base_accs, 'ro', label="Base accs")
    plt.title("Test Task " + str(i) + ": Distance Vs Acc")
    plt.xlabel("Test and Root Task Distance")
    plt.ylabel("Acc")
    plt.legend()
    name = 'graphs/task' + str(i) + "/Distance_Vs_Acc"

    if os.path.isfile(name):
           os.remove(name)
    plt.savefig(name)
    plt.close()


for i, m in enumerate(mamls):
    if not os.path.exists("graphs/model" + str(i)):
        os.mkdir("graphs/model"+str(i))
    epochs = np.arange(m.num_of_epochs)
    plt.plot(epochs, m.avg_train_losses, 'b-', label='Average Train Loss')
    plt.plot(epochs, m.avg_val_losses, 'r-', label='Average Validation Loss')
    plt.title("Model "+str(i) + ": Train Loss vs Val Loss ")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    name = 'graphs/model'+str(i)+'/Train_Loss_Vs_Val_Loss.png'
    if os.path.isfile(name):
           os.remove(name)
    plt.savefig(name)
    plt.close()

    plt.plot(epochs, m.avg_train_accs, 'b-', label='Average Train Acc')
    plt.plot(epochs, m.avg_val_accs, 'r-', label='Average Validation Acc')
    plt.title("Model " + str(i) + ": Train Acc vs Val Acc")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    name = 'graphs/model'+str(i)+'/Train_Acc_Vs_Val_Acc.png'
    if os.path.isfile(name):
           os.remove(name)
    plt.savefig(name)
    plt.close()

if not os.path.exists("graphs/final"):
    os.mkdir("graphs/final")

test_task_nums = np.arange(len(test_tasks))
print(test_accs)
plt.plot(test_task_nums, s_test_accs, 'bo', label='S Model Acc')
plt.plot(test_task_nums, s_base_test_accs, 'ro', label='Base Model Acc')
plt.title("S Model vs Base Model Accs on Test Tasks")
plt.ylabel("Accuracy")
plt.xlabel("Test Task Number")
plt.legend()
name = "graphs/final/final_acc.png"
if os.path.isfile(name):
       os.remove(name)
plt.savefig(name)

print("Mean Accs S: ", np.mean(test_accs))
print("Mean Accs Base:", np.mean(base_test_accs))

if not os.path.exists("graphs/base_model"):
    os.mkdir("graphs/base_model")
epochs = np.arange(m.num_of_epochs)
plt.plot(epochs, base_maml.avg_train_losses, 'b-', label='Average Train Loss')
plt.plot(epochs, base_maml.avg_val_losses, 'r-', label='Average Validation Loss')
plt.title("Base Model: Train Loss vs Val Loss ")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
name = 'graphs/base_model/Train_Loss_Vs_Val_Loss.png'
if os.path.isfile(name):
       os.remove(name)
plt.savefig(name)
plt.close()











