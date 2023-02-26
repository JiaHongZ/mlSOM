import numpy as np
import matplotlib.pyplot as plt
import torch
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def location2matrix(location, row, col):
    '''
    :param location: b, node_num, coordinate
    :return: b, row, col
    '''
    b,n,o = location.shape
    matrix = torch.zeros(b,row,col)
    for i in range(b):
        loc = location[i,:,:]
        # print(matrix.shape,loc)
        matrix[i][loc[:,0].type(torch.long),loc[:,1].type(torch.long)] = 1
        # print(matrix[i])
    return matrix
def getWeight(location, som, input_size, device):
    '''
    改良版，去掉了batch循环维度,效率提升了batch倍
    :param location: b, node_num, coordinate
    :return: b, row, col
    '''
    b,n,o = location.shape
    weights = torch.zeros(n,b,som.input_size)

    for n_index in range(n):
        loc = location[:,n_index,:]
        # print(som.weight.shape)
        weight = som.weight[:,loc[:,0].type(torch.long)*som.output_size[0]+loc[:,1].type(torch.long)].cpu()
        # print(weight.shape)
        # 这里出了问题，不能这样
        # weight = weight.reshape(b,som.input_size)
        # 应该是
        weight = weight.transpose(0,1)
        weights[n_index] = weight
    weights = torch.mean(weights,0)
    return weights

def getWeight_vision1(location, som, input_size, device):
    '''
    :param location: b, node_num, coordinate
    :return: b, row, col
    '''
    b,n,o = location.shape
    weights = torch.zeros(b,n,som.input_size)
    for n_index in range(n):
        for i in range(b):
            loc = location[i,n_index,:]
            # print(som.weight.shape, 49, 1600)
            weight = som.weight[:,loc[0].type(torch.long)*som.output_size[0]+loc[1].type(torch.long)].cpu()
            # print(weight.shape)
            weight = weight.reshape(1,som.input_size)
            weights[i,n_index] = weight
    weights = torch.sum(weights,1) / n
    return weights

def map_show(location,Y):
    # location = location.cpu()
    img = np.zeros((80,80))
    b,n,o = location.shape
    for i in range(b):
        if Y[i] == 0:
            point = location[i]
            for j in range(len(point)):
                img[int(point[j][0])][int(point[j][1])] = 255.0
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # a = torch.zeros([5, 5])
    # index = (torch.LongTensor([0, 1]), torch.LongTensor([1, 2]))
    # print(index[0])
    # value = torch.Tensor([1, 1])  # 生成要填充的值
    # a.index_put_(index, value)
    # print(a)

    a = torch.rand((3,4))
    # b = torch.tensor([ # 1,3
    #     [0,0],
    #     [1,1],
    # ])
    print(a)
    # c = a[:, b[:,0]*1+b[:,1]]
    print('transpose',a.transpose(0,1))
    print('reshape',a.reshape(4,3))
