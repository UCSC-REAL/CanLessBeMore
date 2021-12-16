import os, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from cleanlab.noise_generation import generate_noisy_labels, generate_noise_matrix_from_trace
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.latent_estimation import estimate_noise_matrices
from sklearn.linear_model import LogisticRegression

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_noise', type=float, default=0.4, help='foo help')
parser.add_argument('--min_noise', type=float, default=0.2, help='foo help')
args = parser.parse_args()

def count_agreement(Y, indice, group=0):
    return np.mean(np.array(
        [np.all(Y[i] == Y[indice[i]]) for i in range(Y.size) if Y[i] == group]))

def calculate_agreement_difference(Y, indice):
    return count_agreement(Y, indice, group=1) - count_agreement(Y, indice, group=0)

def monoflip(Y, noise_rate=0.05):
    is_flip = np.random.binomial(1, noise_rate, Y.size)
    return is_flip * (1. - Y) + (1. - is_flip) * Y

def increase_noise(Y, epsilon, group=0):
    Y_new = np.copy(Y)
    Y_new[Y==group] = monoflip(Y_new[Y==group], noise_rate=epsilon)
    return Y_new

def tune_balance(X, Y, indice, threshold=0.001):
    epsilon_l, epsilon_r = 0.0, 0.3
    Y_l, Y_r = np.copy(Y), increase_noise(Y, epsilon_r)
    diff_l, diff_r = calculate_agreement_difference(Y_l, indice), calculate_agreement_difference(Y_r, indice)
    while diff_l > threshold and diff_r < -threshold:
        epsilon_mid = (epsilon_l + epsilon_r) / 2
        Y_mid = increase_noise(Y, epsilon_mid)
        diff_mid = calculate_agreement_difference(Y_mid, indice)

        if diff_mid < - threshold:
            epsilon_r, diff_r = epsilon_mid, diff_mid
        elif diff_mid > threshold:
            epsilon_l, diff_l = epsilon_mid, diff_mid
        else:
            return Y_mid
    return Y_l

def generate_noise_matrix_from_diagonal(diag):
    K = diag.shape[0]
    noise_matrix = np.zeros((K, K))
    for i in range(diag.shape[0]):
        noise_matrix[i, i] = diag[i]
        noise_matrix[np.arange(K)!=i, i] = np.random.dirichlet(np.ones(K-1)) * (1 - diag[i])
    return noise_matrix

# load training data
with open('data/fairface_50dim_train.npy', 'rb') as f:
    X_train= np.load(f)
    Y_train = np.load(f)

with open('data/fairface_50dim_test.npy', 'rb') as f:
    X_test= np.load(f)
    Y_test = np.load(f)

INDICE_PATH = 'data/fairface_indice.npy'
with open(INDICE_PATH, 'rb') as f:
    indice = np.load(f)

noise_matrix = generate_noise_matrix_from_diagonal(np.array([1 - args.min_noise, 1 - args.max_noise]))
Y_noise = generate_noisy_labels(Y_train, noise_matrix)
Y_upsample = tune_balance(X_train, Y_noise, indice)

# learning with noisy labels
estimated_noise_matrix, _ = estimate_noise_matrices(X_train, Y_noise)
misspecified_noise_matrix = generate_noise_matrix_from_trace(2, (2 - args.min_noise - args.max_noise), py=np.array([np.mean(Y_train == k) for k in range(2)]))
sample_weight = np.ones(np.shape(Y_noise))
for k in range(2):
    sample_weight_k = 1.0 / misspecified_noise_matrix[k][k]
    sample_weight[Y_noise == k] = sample_weight_k
clf = LogisticRegression().fit(X_train, Y_noise, sample_weight=sample_weight)
Y_pred = clf.predict(X_test)
misspecified_surrogate_result = (Y_pred == Y_test).mean()*100
print(f"Misspecified Surrogate Loss accuracy: {misspecified_surrogate_result:.2f}")

for k in range(2):
    sample_weight_k = 1.0 / estimated_noise_matrix[k][k]
    sample_weight[Y_noise == k] = sample_weight_k
clf = LogisticRegression().fit(X_train, Y_noise, sample_weight=sample_weight)
Y_pred = clf.predict(X_test)
estimated_surrogate_result = (Y_pred == Y_test).mean()*100
print(f"Estimated Surrogate Loss accuracy: {estimated_surrogate_result:.2f}")
X_train, X_val, Y_noise_train, Y_noise_val, Y_upsample_train, Y_upsample_val = map(torch.from_numpy, train_test_split(X_train, Y_noise, Y_upsample, test_size=0.1, shuffle=True))
X_test, Y_test = torch.from_numpy(X_test), torch.from_numpy(Y_test)

# data loaders
train_dataloader_noise = DataLoader(TensorDataset(X_train, Y_noise_train), batch_size=Y_noise_train.size)
train_dataloader_balance = DataLoader(TensorDataset(X_train, Y_upsample_train), batch_size=Y_upsample_train.size)
val_dataloader_noise = DataLoader(TensorDataset(X_val, Y_noise_val), batch_size=Y_noise_val.size)
val_dataloader_balance = DataLoader(TensorDataset(X_val, Y_upsample_val), batch_size=Y_upsample_val.size)
test_dataloader = DataLoader(TensorDataset(X_test, Y_test))

def run(train_dataloader, val_dataloader, test_dataloader, nFeatures=50):
    prober = torch.nn.Linear(nFeatures, 2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(prober.parameters(), max_iter=50)
    peer_results = 0.
    ce_results = 0.
    for alpha in np.arange(-0.8, 0.85, 0.05):
        prober.reset_parameters()
        
        best_val_accuracy = 0.0
        # train
        for data, target in train_dataloader:
            data, target = data.float(), target.long()
            def closure():
                optimizer.zero_grad()
                output = prober(data.float())
                loss = criterion(output, target) - alpha * criterion(output, target[torch.randperm(target.shape[0])])
                loss.backward()
                return loss
            optimizer.step(closure)     
        # val  
        def evaluate(data_loader):
            correct, total = 0., 0.
            for data, label in data_loader:
                data, label = data.float(), label.long()
                with torch.no_grad():
                    output = prober(data)
                    _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            return 100 * correct / total

        val_accuracy = evaluate(val_dataloader)
        if val_accuracy > best_val_accuracy:
            peer_results = evaluate(test_dataloader)
            best_val_accuracy = val_accuracy

        if abs(alpha) < 0.01:
            ce_results = evaluate(test_dataloader)
    return ce_results, peer_results

ce_result1, peer_result1 = run(train_dataloader_noise, val_dataloader_noise, test_dataloader, nFeatures=X_test.shape[1])
ce_result2, peer_result2 = evaluate(train_dataloader_balance, test_dataloader_balance, nFeatures=X_test.shape[1])

with open(f'log/fairface50/result_{int(args.min_noise*10)}_{int(args.max_noise*10)}.txt', 'a') as f:
    f.write(f"{misspecified_surrogate_result},{estimated_surrogate_result},{ce_result1},{peer_result1},{ce_result2},{peer_result2}\n")