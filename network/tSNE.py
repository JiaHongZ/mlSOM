# 使用vgg预训练模型做迁移学习,vgg19最后一层是1000，所以冻结其他层，只训练新分类层，进行cifar分类，
from collections import namedtuple
import torch
from torch.autograd import Variable
import os
from torchvision.transforms import ToPILImage
import torchvision.models as models
show = ToPILImage()
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from sklearn import manifold, datasets
from network.som_multilayer import SOM
from network.feature_dataloader import Dataset
from network.classifier_3layer_dataloader import Dataset as Dataset_classifier
import tqdm
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.utils import save_image
import network.config as conf
import os
import pickle
import network.utils as utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

DATA_DIR = 'datasets/mnist'
transform = transforms.Compose(
    [transforms.ToTensor()
        , transforms.Normalize(mean=[0], std=[1])]
)
def test_tSNE():
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=300, shuffle=False)
    for i, (img, lab) in enumerate(test_loader):
        b,c,h,w = img.shape
        lab = np.array(lab)  # 转为数字
        # img = Variable(img).cuda()
        X = img.reshape(b, 28*28)
        y = lab.reshape(b)
        break
    # ----------------------------------------------------------------------
    from matplotlib import cm

    def plot_with_labels(lowDWeights, labels, i):
        plt.cla()
        # 降到二维了，分别给x和y
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        # 遍历每个点以及对应标签
        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 / 9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
            plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min()-3, X.max()+3)
        plt.ylim(Y.min()-3, Y.max()+3)
        plt.title('Visualize last layer')
        # plt.savefig("{}.jpg".format(i))
    # ----------------------------------------------------------------------
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # 把64维降到2维

    X_tsne = tsne.fit_transform(X)
    # X_tsne是(1083,2)
    print(X_tsne.shape)

    # plot_embedding(X_tsne,"t-SNE embedding of the digits")
    plot_with_labels(X_tsne,y,"t-SNE embedding of the digits")
    plt.show()

if __name__ == '__main__':
    test_tSNE()