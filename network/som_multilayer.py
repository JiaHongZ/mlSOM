import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.nn import functional as F

import numpy as np
class SOM(nn.Module):
    def __init__(self, input_size, output_size=(10, 10), lr=0.3, sigma=None, n=1, code_num=10, kernel_size=(7,7), stride=7, kernel_output_size=(28,28)):
        '''
        :param input_size:
        :param output_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n = n
        self.lr = lr
        self.code_num = code_num
        if sigma is None:
            self.sigma = 2
        else:
            self.sigma = sigma
        self.weight = nn.Parameter(torch.randn(input_size, output_size[0] * output_size[1]), requires_grad=False)
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=0).cuda()
        self.fold = torch.nn.Fold(output_size=kernel_output_size, kernel_size=kernel_size, stride=stride, padding=0).cuda()

    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.output_size[0]):
            for y in range(self.output_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location of best matching unit, loss
        '''
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        dists = torch.sort(dists, dim=1, descending=False, out=None)

        bmu_indexes = dists.indices[:,:self.n]
        bmu_locations = self.locations[bmu_indexes]
        return bmu_locations.transpose(0,1), 0

    def getWeight(self, location):
        b, n, o = location.shape
        weights = torch.zeros(n, b, self.input_size)
        for n_index in range(n):
            loc = location[:, n_index, :]
            weight = self.weight[:, loc[:, 0].type(torch.long) * self.output_size[0] + loc[:, 1].type(torch.long)].cpu()
            weight = weight.transpose(0, 1)
            weights[n_index] = weight
        weights = torch.mean(weights, 0)
        return weights

    def self_organizing(self, input, current_iter, max_iter):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction

        #Find best matching unit
        bmu_locations, loss = self.forward(input)

        delta_ = torch.zeros_like(self.weight)
        distance_square = self.locations.float() - bmu_locations[0].unsqueeze(1).float()
        distance_square.pow_(2)
        distance_square = torch.sum(distance_square, dim=2)
        lr_locations = self._neighborhood_fn(distance_square, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)
        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        delta_ += delta
        for i in range(1, len(bmu_locations)):
            distance_square = self.locations.float() - bmu_locations[i].unsqueeze(1).float()
            distance_square.pow_(2)
            distance_square = torch.sum(distance_square, dim=2)
            lr_locations = self._neighborhood_fn(distance_square, sigma)
            lr_locations.mul_(lr).unsqueeze_(1)

            delta = lr_locations * (input.unsqueeze(2) - self.weight)
            delta = delta.sum(dim=0)
            delta.div_(batch_size)
            delta = delta/i
            delta_ += delta
        self.weight.data.add_(delta_)
        # print(torch.sum(x-self.weight.data))
        return loss
    def patch_self_organizing(self, input, epo, epoch):
        patches = self.unfold(input)
        patches = patches.permute(2, 0, 1)  # (patch number, b, 7*7)
        for i_patch in range(len(patches)):
            loss = self.self_organizing(patches[i_patch], epo, epoch)  # train som
        return loss
    def restore(self, X):
        b,c,h,w = X.shape
        patches = self.unfold(X)
        patches = patches.permute(2, 0, 1)  # (patch number, b, 7*7)
        for i_patch in range(len(patches)):
            location1 = self.map_multivects(patches[i_patch])  # batch, node number, coordinate
            patches[i_patch] = self.getWeight(location1).cuda()
        patches = patches.permute(1, 2, 0)
        inputs_restore = self.fold(patches)  # (b,c,h,w)
        inputs_restore = inputs_restore.reshape(b, c, h, w)
        return inputs_restore
    def location2matrix(self,location):
        b, n, o = location.shape
        matrix = torch.zeros(b, self.output_size[0], self.output_size[1])
        for i in range(b):
            loc = location[i, :, :]
            matrix[i][loc[:, 0].type(torch.long), loc[:, 1].type(torch.long)] = 1
        return matrix
    def get_feature(self, X):
        b,c,h,w = X.shape
        patches = self.unfold(X)
        patches = patches.permute(2, 0, 1)  # (patch number, b, 7*7)
        matrix_som1 = torch.zeros(b, self.output_size[0], self.output_size[1])
        for i_patch in range(len(patches)):
            location1 = self.map_multivects(patches[i_patch])  # batch, node number, coordinate
            matrix1_som = self.location2matrix(location1)
            matrix_som1 = matrix1_som + matrix_som1
            patches[i_patch] = self.getWeight(location1)
        matrix_som1[matrix_som1 > 1] = 1
        matrix_som1[matrix_som1 < 1] = 0
        X1 = matrix_som1.contiguous().cuda()

        patches = patches.permute(1, 2, 0)
        inputs_restore = self.fold(patches)  # (b,c,h,w)
        inputs_restore = inputs_restore.reshape(b, c, h, w)
        return X1, inputs_restore

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.output_size[0] * self.output_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.output_size[0])

    def map_vects(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_locations = self.locations[bmu_indexes]

        return bmu_locations

    def map_multivects(self, input):
        n = self.code_num
        batch_size = input.size()[0]
        input = input.view(batch_size, -1, 1)
        batch_weight = self.weight.expand(batch_size, -1, -1)

        dists = self.pdist_fn(input, batch_weight)
        dist = torch.sort(dists, dim=1, descending=False, out=None)
        # Find best matching unit
        # losses, bmu_indexes = dists.min(dim=1, keepdim=True)
        bmu_indexes = dist.indices[:,:n]
        bmu_locations = self.locations[bmu_indexes]
        return bmu_locations

class Hebb(nn.Module):
    def __init__(self, input_size, output_size=10, lr=0.3):
        '''
        :param input_size:
        :param output_size:
        :param lr:
        :param sigma:
        '''
        super(Hebb, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.inhibit = 0
        self.lr = lr
        self.tau = 1 # 温度常数
        self.weight = nn.Parameter(torch.zeros(input_size, output_size), requires_grad=False)
        # self.weight = nn.Parameter(torch.randn(input_size, output_size), requires_grad=False)

    def step(self, X, Y):
        X = X.flatten(1)
        b, _ = X.shape
        deltaW = self.lr * (torch.mm(X.T,Y) - self.inhibit)/b
        # print(list(deltaW))
        self.weight.data += deltaW
        # loss
        out = torch.mm(X, self.weight.data)
        _, lab_pre = torch.max(out.data, 1)

        lab_pre = lab_pre.unsqueeze(1).cpu()
        one_hot = torch.zeros(b, self.output_size).scatter_(1, lab_pre, 1)
        return (one_hot-Y.cpu())/b

    def forward(self, X):
        X = X.flatten(1)
        b, _ = X.shape
        out = torch.mm(X, self.weight)
        return out