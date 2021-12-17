import torch
import torch.nn

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import numpy as np

import matplotlib.pyplot as plt # plotting library
import pandas as pd # this module is useful to work with tabular data
#import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(num_features=4),
            nn.ReLU(True),
            nn.Linear(4, 2),
            nn.BatchNorm1d(num_features=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(num_features=4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.BatchNorm1d(num_features=8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

def extract_action(tensor_data):
    numpy_data = tensor_data.numpy()[0]
    action_data = numpy_data[-3:]

    return np.argmax(action_data)

#  Hyper Parameter 설정
num_epochs = 100
batch_size = 512
learning_rate = 1e-3

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size = 1, shuffle=False)

#  모델 설정
encoder = encoder().cuda().train()
decoder = decoder().cuda().train()

#  모델 Optimizer 설정
criterion = nn.MSELoss()
criterion_2 = nn.CrossEntropyLoss()
encoder_optimizer = torch.optim.Adam( encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
decoder_optimizer = torch.optim.Adam( decoder.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    loss_graph = []
    loss_mse_graph = []
    loss_cross_graph = []

    vis_flag = True

    encoder.train()
    decoder.train()

    for data in dataloader:
        data = Variable(data).float().cuda()
        # ===================forward=====================
        latent_z = encoder(data)
        output = decoder(latent_z)
        # ===================backward====================
        #print(data[:,-3:].long())
        loss_mse = criterion(output[:,:-3], data[:, :-3])
        loss_cross = criterion_2(output[:,-3:], torch.argmax(data[:,-3:].long(), dim=1))

        loss = loss_mse + loss_cross

        loss_graph.append(loss.item())
        loss_mse_graph.append(loss_mse.item())
        loss_cross_graph.append(loss_cross.item())

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    print("MSE : {} Cross : {}".format(np.mean(loss_mse_graph), np.mean(loss_cross_graph)))
    print("Epoch : {} Loss : {}".format(epoch, np.mean(loss_graph)))

    encoder.eval()

    encoded_samples = []

    print(len(test_dataloader))

    for data in dataloader:
        label = extract_action(data)
        data = Variable(data).float().cuda()
        # ===================forward=====================
        with torch.no_grad():
            latent_z = encoder(data)

            encoded_sample = latent_z.cpu().numpy()
            encoded_sample = {f"Variable {i}": enc for i, enc in enumerate(encoded_sample[0])}
            encoded_sample['label'] = label

            encoded_samples.append(encoded_sample)

    encoded_samples = pd.DataFrame(encoded_samples)
    print(encoded_samples)

    plt.figure(figsize=(17, 9))
    plt.scatter(encoded_samples['Variable 0'], encoded_samples['Variable 1'], c=encoded_samples.label, cmap='tab10')
    plt.colorbar()
    plt.savefig('figure/' + str(epoch) + '.png')
