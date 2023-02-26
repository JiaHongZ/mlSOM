'''
    multiple layers patch-based SOM
'''
from network.som_multilayer import SOM
import tqdm
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn as nn
from torchvision.utils import save_image
import network.config as conf
import os
import pickle
import network.utils as utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn import manifold, datasets

# 逐层训练
class mpSOM(nn.Module):
    def __init__(self, conf):
        super(mpSOM, self).__init__()
        self.layer1_conf = conf['layer1']

        self.save_path = conf['save_path']
        self.layer1 = SOM(input_size=self.layer1_conf['input_size'], output_size=self.layer1_conf['output_size'],
                          lr=self.layer1_conf['lr'],sigma=self.layer1_conf['sigma'],n=self.layer1_conf['n'],
                          code_num=self.layer1_conf['code_num'], kernel_size=self.layer1_conf['kernel_size']
                          , stride=self.layer1_conf['stride'], kernel_output_size=self.layer1_conf['kernel_output_size']
                          )

        now_path = os.getcwd()
        self.save_path = os.path.join(now_path, self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.layer1_feature_train = os.path.join(self.save_path, 'layer1_feature', 'train')
        self.layer1_feature_test = os.path.join(self.save_path, 'layer1_feature', 'test')
        self.layer1_pth = os.path.join(self.save_path,'layer1.pth')

        if not os.path.exists(self.layer1_feature_train):
            os.makedirs(self.layer1_feature_train)
            os.makedirs(self.layer1_feature_test)

    def train_layer1(self, epoch=200, train_data=None, test_data=None, device=None, train=True):
        if train:
            for epo in range(epoch):
                print('layer1,epoch', epo)
                for idx, (X, Y) in enumerate(train_loader):
                    X = X.cuda()
                    self.layer1.patch_self_organizing(X, epo, epoch)  # train som
                torch.save(self.layer1.state_dict(), self.layer1_pth)
                if epo % 10 == 0:
                    if not os.path.exists(os.path.join(self.save_path,'layer1')):
                        os.makedirs(os.path.join(self.save_path,'layer1'))
                    self.layer1.save_result(os.path.join(self.save_path,'layer1','%d_layer1_result.png' % (epo)), (1, 14, 14))
            torch.save(self.layer1.state_dict(), self.layer1_pth)
        else:
            self.layer1.load_state_dict(torch.load(self.layer1_pth), strict=True)

    def load(self):
        self.layer1.load_state_dict(torch.load(self.layer1_pth), strict=True)

    def run(self, dataloader, classifer):
        self.load()
        currect = 0
        count = 0
        for idx, (X, Y) in enumerate(tqdm.tqdm(dataloader)):
            # X[:, :, 7:21, 7:21] = 0
            b, c, h, w = X.shape
            X = X.cuda()
            # layer 1
            X1, inputs_restore = self.layer1.get_feature(X)
            # classifier
            matrix = X1.flatten(1)
            b, _ = matrix.shape
            Y = Y.squeeze().cuda()
            with torch.no_grad():
                pre = classifer.classifier(matrix)
            _, lab_pre = torch.max(pre.data, 1)
            currect += torch.sum(lab_pre == Y.data)
            count += b
        current_rate = torch.true_divide(currect, count)
        print('acc:', current_rate)
        return current_rate

    def show_part(self):
        for idx, (X, Y) in enumerate(train_loader):
            plt.imshow(X.squeeze().cpu().numpy())
            plt.show()
            X = X.cuda()
            X1, inputs_restore = self.layer1.get_feature(X)
            plt.imshow(X1.squeeze().cpu().numpy())
            plt.show()

            images = self.layer1.weight.view(1, 14, 14, 44 * 44)
            images = images.permute(3, 0, 1, 2)
            # for i in range(44):
            #     for j in range(44):
            #         if X1.squeeze()[i][j] == 0:
            #             images[i*44+j] = torch.zeros(1,14,14)
            save_image(images, 'activity.png', normalize=True, padding=1, nrow=44)
            break

    def show_cluster(self):
        from matplotlib import cm
        import numpy as np
        def plot_with_labels(lowDWeights, labels, i):
            plt.cla()
            fig = plt.figure()
            ax = fig.add_subplot(111)  # 创建子图
            ax.set
            # 降到二维了，分别给x和y
            X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
            # 遍历每个点以及对应标签
            count = [1] * 10
            sign_count = 50
            for x, y, s in zip(X, Y, labels):
                c = cm.rainbow(int(255 / 9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
                if count[s] == 1:
                    if s == 5:
                        plt.text(-25, -30, int(s + 1), backgroundcolor=c, fontsize=12)
                    elif s == 8:
                        plt.text(-30, 0, int(s + 1), backgroundcolor=c, fontsize=12)
                    else:
                        plt.text(x, y, int(s+1), backgroundcolor=c, fontsize=12)
                    count[s] -= 1
                ax.scatter(x, y, c=c, s=5, cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
            # plt.axis('off')
            plt.xlim(X.min()-5, X.max()+5)
            plt.ylim(Y.min()-5, Y.max()+5)
            # plt.title('Visualize last layer')

        # 每个类选20个样本
        box = [100] * 10

        X_arr = []
        Y_arr = []
        for idx, (X, Y) in enumerate(test_loader):
            if box[Y] == 0:
                continue
            if sum(box) == 0:
                break
            box[Y] -= 1
            X = X.cuda()
            # layer 1
            X1, inputs_restore = self.layer1.get_feature(X)
            # classifier
            matrix = X1.flatten(1)

            if idx == 0:
                X_arr = matrix.cpu()
                Y_arr = Y.unsqueeze(0).cpu()
            else:
                X_arr = torch.cat([X_arr, matrix.cpu()], 0)
                Y_arr = torch.cat([Y_arr, Y.unsqueeze(0).cpu()], 0)

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        # print(X_arr.shape, Y_arr.shape)
        # print(Y_arr)
        X_tsne = tsne.fit_transform(X_arr)
        # print(X_tsne)
        # print(X_tsne.shape)
        # plot_embedding(X_tsne,"t-SNE embedding of the digits")
        plot_with_labels(X_tsne, Y_arr, "t-SNE embedding of the digits")
        plt.show()

if __name__ == '__main__':
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = 'datasets/mnist'
    config = conf.mpSOM_MNIST_1layers_patch_overlap
    transform = transforms.Compose(
        [transforms.ToTensor()
        , transforms.Normalize(mean=[0], std=[1])]
    )
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    # train_data.data = train_data.data[0:1024]
    # test_data.data = train_data.data[0:1024]
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    mpsom = mpSOM(config).cuda()
    mpsom.load()
    # weight = mpsom.layer1.weight
    # print(weight.shape)
    # mpsom.layer1.weight.view(44,44,14*14)

    # mpsom.show_part()
    mpsom.show_cluster()