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
            print(lab_pre,Y)
            currect += torch.sum(lab_pre == Y.data)
            count += b
        current_rate = torch.true_divide(currect, count)
        print('acc:', current_rate)
        return current_rate

    def show_part(self):
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image

        for idx, (X, Y) in enumerate(test_loader):
            if Y[0] == 9 or Y[0] == 7 or Y[0] == 4:
                plt.imshow(X.squeeze().cpu().numpy())
                plt.savefig(str(Y[0])+'_'+str(idx)+'org.png')
                # plt.show()

                # plt.show()
                X = X.cuda()
                X1, inputs_restore = self.layer1.get_feature(X)

                plt.imshow(X1.squeeze().cpu().numpy())
                plt.savefig(str(Y[0])+'_'+str(idx)+'feature.png')
                # plt.show()

                images = self.layer1.weight.view(1, 14, 14, 44 * 44)

                images = images.permute(3, 0, 1, 2)
                for i in range(44):
                    for j in range(44):
                        if X1.squeeze()[i][j] == 0:
                            images[i*44+j] = torch.zeros(1,14,14)
                save_image(images, str(Y[0])+'_'+str(idx)+'activity.png', normalize=True, padding=1, nrow=44)
            # break

    def show_part_overleap(self, classifer):
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image
        X1_sum = 0
        images_sum = 0
        target = 4
        for idx, (X, Y) in enumerate(test_loader):
            if idx == 299:
                break
            if Y[0] == target:
                X = X.cuda()
                X1, inputs_restore = self.layer1.get_feature(X)

                images = self.layer1.weight.view(1, 14, 14, 44 * 44)

                matrix = X1.flatten(1)
                b, _ = matrix.shape
                Y = Y.squeeze().cuda()
                with torch.no_grad():
                    pre = classifer.classifier(matrix)
                _, lab_pre = torch.max(pre.data, 1)

                # 只算分类对的
                # print(Y,lab_pre[0])
                if lab_pre[0] == Y:
                    images = images.permute(3, 0, 1, 2)
                    for i in range(44):
                        for j in range(44):
                            if X1.squeeze()[i][j] == 0:
                                images[i * 44 + j] = torch.zeros(1, 14, 14)

                    X1_sum += X1
                    images_sum += images
            # break

        plt.imshow(X1_sum.squeeze().cpu().numpy())
        plt.savefig(str(target) + '_' + str('sum') + 'feature.png')
        save_image(images_sum, str(target) + '_' + str('sum') + 'activity.png', normalize=False, padding=1, nrow=44)

    def getX_I(self, target, classifer):
        for idx, (X2, Y2) in enumerate(test_loader):
            if Y2[0] == target:
                X2 = X2.cuda()
                X2, inputs_restore = self.layer1.get_feature(X2)
                images2 = self.layer1.weight.view(1, 14, 14, 44 * 44)
                matrix2 = X2.flatten(1)
                b, _ = matrix2.shape
                Y2 = Y2.squeeze().cuda()
                with torch.no_grad():
                    pre2 = classifer.classifier(matrix2)
                _, lab_pre2 = torch.max(pre2.data, 1)

                # 只算分类对的
                print(Y2,lab_pre2[0])
                if lab_pre2[0] == Y2:
                    images2 = images2.permute(3, 0, 1, 2)
                    for i in range(44):
                        for j in range(44):
                            if X2.squeeze()[i][j] == 0:
                                images2[i * 44 + j] = torch.zeros(1, 14, 14)
                    break
        return X2, images2
    def show_part_overleap2(self, classifer):

        target1 = 4
        target2 = 7
        X1, images1 = self.getX_I(target2, classifier)
        X2, images2 = self.getX_I(target1, classifier)
        print(images1.shape, images2.shape)
        plt.imshow((X1*50 + X2*100).squeeze().cpu().numpy(),cmap='rainbow')
        plt.savefig(str(target1) + '_' + str(target2)  + '_' + str('sum') + 'feature.png')
        save_image(images2+images1, str(target1) + '_' + str(target2) + '_' + str('sum') + 'activity.png', normalize=False, padding=1, nrow=44)

class SL(torch.nn.Module):
    def __init__(self, hidden, classese):
        super().__init__()
        self.linear1 = nn.Linear(hidden, classese)
        # self.relu = nn.ReLU()
        # self.drop = nn.Dropout(0.2)
        # self.linear2 = nn.Linear(256, classese)
        self.softmax = nn.Softmax()
    def forward(self, x):
        # x = torch.flatten(x,1)
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.drop(x)
        # x = self.linear2(x)
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
            # classifier
            matrix = X1.flatten(1)
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
        optimiter = torch.optim.Adam(self.classifier.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8,
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
                # classifier
                matrix = X1.flatten(1)
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
            if epo % 10 == 0:
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
    mpsom2 = mpSOM(config).cuda()
    mpsom2.load()

    classifier = Classifier(config).cuda()
    classifier.load(mpsom.save_path)

    target1 = 7
    target2 = 9
    X1, images1 = mpsom.getX_I(target2, classifier)
    X2, images2 = mpsom2.getX_I(target1, classifier)
    print(images1.shape, images2.shape)
    plt.imshow((X1*10 + X2*50).squeeze().cpu().numpy(),cmap='rainbow')
    plt.savefig(str(target1) + '_' + str(target2)  + '_' + str('sum') + 'feature.png')
    save_image(images2+images1, str(target1) + '_' + str(target2) + '_' + str('sum') + 'activity.png', normalize=False, padding=1, nrow=44)
