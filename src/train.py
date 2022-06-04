import mask_dataset
from config import config
from model import MaskCNN

import os
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = mask_dataset.MaskDataset(
    root_dir='dataset', img_dim=config.img_dim)

train_loader, valid_loader = dataset.get_data_loaders(
    batch_size=config.batch_size, use_shuffle=config.use_shuffle, split_ratio=config.split_ratio)

model = MaskCNN(len(dataset.labels), config.img_dim,
                base_filter_size=config.base_filter_size)
model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

avg_loss = 0
best_vloss = 999999

for epoch in range(config.epochs):
    # traning
    model.train(True)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        out_labels = model(images)
        loss = loss_fn(out_labels, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print every config.log_interval mini-batches
        if i % config.log_interval == config.log_interval - 1:
            avg_loss = running_loss / config.log_interval
            print(
                f'[epoch_{epoch + 1}, batch_{i + 1}]\ttrain_loss: {avg_loss:.3f}')
            running_loss = 0.0

    # validation
    model.train(False)
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(valid_loader):
            vimages, vlabels = vdata[0].to(device), vdata[1].to(device)
            vout_labels = model(vimages)
            vloss = loss_fn(vout_labels, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print(
        f'[epoch_{epoch + 1} ended]\ttrain_loss: {avg_loss:.3f}, valid_loss: {avg_vloss:.3f}\n')

    # save the best model.
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), f'../trained/MaskCNN_0.pt')

print('\ndone.')

model.load_state_dict(torch.load('../trained/MaskCNN_0.pt'))
model.eval()

total = 0
correct = 0

total_per_class = [0, 0, 0, 0, 0]
correct_per_class = [0, 0, 0, 0, 0]

with torch.no_grad():
    for data in valid_loader:
        images, labels = data[0].to(device), data[1].to(device)

        out_labels = model(images)
        preds = torch.argmax(out_labels.data, dim=1)
        total += labels.shape[0]
        correct += (preds == labels).sum().item()
        total_per_class[labels[0]] += labels.shape[0]
        correct_per_class[labels[0]] += (preds == labels).sum().item()

total = np.array(total)
correct = np.array(correct)
print(f'val - accuracy: {np.around(100 * correct / total, 2)}%')

total_per_class = np.array(total_per_class)
correct_per_class = np.array(correct_per_class)
acc_per_class = np.around(100 * correct_per_class / total_per_class, 2)

print('val - per class accuracy:')
for idx, acc in enumerate(acc_per_class):
    print(f' - {dataset.label_to_str[idx]}: {acc}%')

print(f'val - avg accuracy: {np.around(acc_per_class.mean(), 2)}%')

total = 0
correct = 0

total_per_class = [0, 0, 0, 0, 0]
correct_per_class = [0, 0, 0, 0, 0]

with torch.no_grad():
    for data in train_loader:
        images, labels = data[0].to(device), data[1].to(device)

        out_labels = model(images)
        preds = torch.argmax(out_labels.data, dim=1)
        total += labels.shape[0]
        correct += (preds == labels).sum().item()
        total_per_class[labels[0]] += labels.shape[0]
        correct_per_class[labels[0]] += (preds == labels).sum().item()

total = np.array(total)
correct = np.array(correct)
print(f'train - accuracy: {np.around(100 * correct / total, 2)}%')

total_per_class = np.array(total_per_class)
correct_per_class = np.array(correct_per_class)
acc_per_class = np.around(100 * correct_per_class / total_per_class, 2)

print('train - per class accuracy:')
for idx, acc in enumerate(acc_per_class):
    print(f' - {dataset.label_to_str[idx]}: {acc}%')

print(f'train - avg accuracy: {np.around(acc_per_class.mean(), 2)}%')
