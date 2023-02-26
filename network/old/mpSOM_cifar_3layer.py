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
        self.layer2_conf = conf['layer2']
        self.layer3_conf = conf['layer3']

        self.save_path = conf['save_path']
        self.layer1 = SOM(input_size=self.layer1_conf['input_size'], output_size=self.layer1_conf['output_size'],
                          lr=self.layer1_conf['lr'],sigma=self.layer1_conf['sigma'],n=self.layer1_conf['n'],
                          code_num=self.layer1_conf['code_num'], kernel_size=self.layer1_conf['kernel_size']
                          , stride=self.layer1_conf['stride'], kernel_output_size=self.layer1_conf['kernel_output_size']
                          )
        self.layer2 = SOM(input_size=self.layer2_conf['input_size'], output_size=self.layer2_conf['output_size'],
                          lr=self.layer2_conf['lr'],sigma=self.layer2_conf['sigma'],n=self.layer2_conf['n'],
                          code_num=self.layer2_conf['code_num'], kernel_size=self.layer2_conf['kernel_size']
                          , stride=self.layer2_conf['stride'], kernel_output_size=self.layer2_conf['kernel_output_size']
                          )
        self.layer3 = SOM(input_size=self.layer3_conf['input_size'], output_size=self.layer3_conf['output_size'],
                          lr=self.layer3_conf['lr'],sigma=self.layer3_conf['sigma'],n=self.layer3_conf['n'],
                          code_num=self.layer3_conf['code_num'], kernel_size=self.layer3_conf['kernel_size']
                          , stride=self.layer3_conf['stride'], kernel_output_size=self.layer3_conf['kernel_output_size']
                          )

        now_path = os.getcwd()
        self.save_path = os.path.join(now_path, self.save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.layer1_feature_train = os.path.join(self.save_path, 'layer1_feature', 'train')
        self.layer1_feature_test = os.path.join(self.save_path, 'layer1_feature', 'test')
        self.layer1_pth = os.path.join(self.save_path,'layer1.pth')

        self.layer2_feature_train = os.path.join(self.save_path, 'layer2_feature', 'train')
        self.layer2_feature_test = os.path.join(self.save_path, 'layer2_feature', 'test')
        self.layer2_pth = os.path.join(self.save_path,'layer2.pth')

        self.layer3_feature_train = os.path.join(self.save_path, 'layer3_feature', 'train')
        self.layer3_feature_test = os.path.join(self.save_path, 'layer3_feature', 'test')
        self.layer3_pth = os.path.join(self.save_path,'layer3.pth')

        if not os.path.exists(self.layer1_feature_train):
            os.makedirs(self.layer1_feature_train)
            os.makedirs(self.layer1_feature_test)
            os.makedirs(self.layer2_feature_train)
            os.makedirs(self.layer2_feature_test)
            os.makedirs(self.layer3_feature_train)
            os.makedirs(self.layer3_feature_test)

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
                    self.layer1.save_result(os.path.join(self.save_path,'layer1','%d_layer1_result.png' % (epo)), (3, 8, 8))
            torch.save(self.layer1.state_dict(), self.layer1_pth)
        else:
            self.layer1.load_state_dict(torch.load(self.layer1_pth), strict=True)

    def train_layer2(self, epoch=150, device=None, train=True):
        if train:
            for epo in range(epoch):
                print('layer2,epoch', epo)
                for idx, (X, Y) in enumerate(train_loader):
                    b, c, h, w = X.shape
                    X = X.cuda()
                    # layer1
                    inputs_restore = self.layer1.restore(X)
                    self.layer2.patch_self_organizing(inputs_restore, epo, epoch)  # train som
                torch.save(self.layer2.state_dict(), self.layer2_pth)
                if epo % 10 == 0:
                    if not os.path.exists(os.path.join(self.save_path,'layer2')):
                        os.makedirs(os.path.join(self.save_path,'layer2'))
                    self.layer2.save_result(os.path.join(self.save_path,'layer2','%d_layer2_result.png' % (epo)), (3, 16, 16))
            torch.save(self.layer2.state_dict(), self.layer2_pth)
        else:
            self.layer2.load_state_dict(torch.load(self.layer2_pth),strict=True)

    def train_layer3(self, epoch=150, device=None, train=True):
        if train:
            for epo in range(epoch):
                print('layer3, epoch', epo)
                for idx, (X, Y) in enumerate(train_loader):
                    b, c, h, w = X.shape
                    X = X.cuda()
                    inputs_restore = self.layer1.restore(X)
                    inputs_restore = self.layer2.restore(inputs_restore)
                    self.layer3.patch_self_organizing(inputs_restore, epo, epoch)  # train som
                torch.save(self.layer3.state_dict(), self.layer3_pth)
                if epo % 10 == 0:
                    if not os.path.exists(os.path.join(self.save_path,'layer3')):
                        os.makedirs(os.path.join(self.save_path, 'layer3'))
                    self.layer3.save_result(os.path.join(self.save_path, 'layer3', '%d_layer3_result.png' % (epo)),
                                            (3, 32, 32))
            torch.save(self.layer3.state_dict(), self.layer3_pth)
        else:
            self.layer3.load_state_dict(torch.load(self.layer3_pth), strict=True)

    def load(self):
        self.layer1.load_state_dict(torch.load(self.layer1_pth), strict=True)
        self.layer2.load_state_dict(torch.load(self.layer2_pth), strict=True)
        self.layer3.load_state_dict(torch.load(self.layer3_pth), strict=True)

    def run(self, dataloader, classifer):
        self.load()
        currect = 0
        count = 0
        for idx, (X, Y) in enumerate(tqdm.tqdm(dataloader)):
            X[:, :, 7:21, 7:21] = 0
            b, c, h, w = X.shape
            X = X.cuda()
            # layer 1
            X1, inputs_restore = self.layer1.get_feature(X)
            # Layer 2
            X2, inputs_restore = self.layer2.get_feature(inputs_restore)
            # Layer 3
            location = self.layer5.map_multivects(inputs_restore)  # train som
            X3 = utils.location2matrix(location, self.layer3_conf['output_size'][0],
                                       self.layer3_conf['output_size'][1]).cuda()  # batch, h, w

            # classifier
            matrix = torch.cat([X1.flatten(1), X2.flatten(1), X3.flatten(1)], 1)
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
class SL(torch.nn.Module):
    def __init__(self, hidden, classese):
        super().__init__()
        self.linear1 = nn.Linear(hidden, 256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, classese)
        self.softmax = nn.Softmax()
    def forward(self, x):
        # x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
class Classifier(torch.nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.conf = conf['classifier']
        self.classifier = SL(self.conf['input_size'], self.conf['output_size'])

    def test(self, sl, mpsom, test_loader):
        sl.eval()
        currect = 0
        count = 0
        for idx, (X, Y) in enumerate(tqdm.tqdm(test_loader)):
            b, c, h, w = X.shape
            X = X.cuda()
            # layer 1
            X1, inputs_restore = mpsom.layer1.get_feature(X)
            # Layer 2
            X2, inputs_restore = mpsom.layer2.get_feature(inputs_restore)
            # Layer 3
            location = mpsom.layer3.map_multivects(inputs_restore)  # train som
            X3 = utils.location2matrix(location, mpsom.layer3_conf['output_size'][0],
                                       mpsom.layer3_conf['output_size'][1]).cuda()  # batch, h, w

            # classifier
            matrix = torch.cat(
                [X1.flatten(1), X2.flatten(1), X3.flatten(1)], 1)
            b, _ = matrix.shape
            Y = Y.squeeze().cuda()
            with torch.no_grad():
                pre = sl(matrix)
            _, lab_pre = torch.max(pre.data, 1)
            currect += torch.sum(lab_pre == Y.data)
            count += b
        sl.train()
        current_rate = torch.true_divide(currect, count)
        print('acc:', current_rate)
        return current_rate

    def train(self, mpsom=None, epoch=100):
        mpsom.load()
        # self.load(mpsom.save_path)
        optimiter = torch.optim.Adam(self.classifier.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0001)
        ######### Scheduler ###########
        warmup = True
        if warmup:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7)
        loss_f1 = torch.nn.CrossEntropyLoss()  # 与pattern的损失函数
        best = 0
        for epo in range(epoch):
            print('classifier', epo)
            current_rate = 0
            count = 0
            for idx, (X, Y) in enumerate(tqdm.tqdm(train_loader)):
                b, c, h, w = X.shape
                X = X.cuda()
                # layer 1
                X1, inputs_restore = mpsom.layer1.get_feature(X)
                # Layer 2
                X2, inputs_restore = mpsom.layer2.get_feature(inputs_restore)
                # Layer 3
                location = mpsom.layer3.map_multivects(inputs_restore)  # train som
                X3 = utils.location2matrix(location, mpsom.layer3_conf['output_size'][0],
                                           mpsom.layer3_conf['output_size'][1]).cuda()  # batch, h, w

                # classifier
                matrix = torch.cat(
                    [X1.flatten(1), X2.flatten(1), X3.flatten(1)], 1)
                b, _ = matrix.shape
                Y = Y.squeeze().cuda()
                pre = self.classifier(matrix)
                loss = loss_f1(pre, Y)
                optimiter.zero_grad()
                loss.backward()
                optimiter.step()

                _, lab_pre = torch.max(pre.data, 1)
                current_rate += torch.sum(lab_pre == Y.data)
                count += b
            acc = torch.true_divide(current_rate, count)
            print('epo', epo, acc)
            if epo % 5 == 0:
                scheduler.step()
                print('epoch', epo, ' current learning rate', optimiter.param_groups[0]['lr'])
                acc = self.test(self.classifier, mpsom ,test_loader)
                if acc > best:
                    torch.save(self.classifier.state_dict(), os.path.join(mpsom.save_path, 'bp.pth'))
                    best = acc
                print('best', best)
        return best
    def load(self,path):
        self.classifier.load_state_dict(torch.load(os.path.join(path,'bp.pth')), strict=True)
if __name__ == '__main__':
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = 'datasets/cifar'
    config = conf.mpSOM_CIFAR_3layers
    transform = transforms.Compose(
        [transforms.ToTensor()
        , transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        ]
    )
    train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    # train_data.data = train_data.data[0:1024]
    # test_data.data = train_data.data[0:1024]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    # train
    mpsom = mpSOM(config).cuda()
    classifier = Classifier(config).cuda()
    mpsom.train_layer1(epoch=200, train_data=train_data, test_data=test_data,device=device,train=False)
    mpsom.train_layer2(epoch=200, train=False,device=device)
    mpsom.train_layer3(epoch=200, train=True,device=device)
    mpsom.load()
    classifier.train(mpsom, 200)

    # test
    mpsom = mpSOM(config).cuda()
    classifier = Classifier(config).cuda()
    mpsom.load()
    classifier.load(mpsom.save_path)
    mpsom.run(test_loader,classifier)