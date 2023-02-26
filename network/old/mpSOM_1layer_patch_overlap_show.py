'''
    multiple layers patch-based SOM
'''
from network.som_multilayer import SOM
from network.feature_dataloader import Dataset
from network.classifier_3layer_dataloader import Dataset as Dataset_classifier
import tqdm
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn as nn
from torchvision.utils import save_image
import network.config as conf
import os
import pickle
import network.utils as utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image

        for idx, (X, Y) in enumerate(train_loader):
            plt.imshow(X.squeeze().cpu().numpy())
            plt.show()
            X = X.cuda()
            X1, inputs_restore = self.layer1.get_feature(X)
            plt.imshow(X1.squeeze().cpu().numpy())
            plt.show()

            images = self.layer1.weight.view(1, 14, 14, 44 * 44)

            images = images.permute(3, 0, 1, 2)
            for i in range(44):
                for j in range(44):
                    if X1.squeeze()[i][j] == 0:
                        images[i*44+j] = torch.zeros(1,14,14)
            save_image(images, 'activity.png', normalize=True, padding=1, nrow=44)

            break
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

    mpsom.show_part()
