input_minst = 28*28
input_cifar = 3*32*32
patch_row_mnist = 2
patch_row_cifar = 4
field_row_mnist_layer2 = 2
field_row_cifar_layer2 = 2

SOM = {
    'save_path': 'model/SOM/',
    'classifier': {
        'input_size': 20 * 20 + 30 * 30 + 44 * 44,
        'output_size': 10,
    }
}
mpSOM_MNIST_1layers_org = {
    'save_path': 'model/mpSOM_MNIST_1layers_orgsom/',
    'layer1':{
        'input_size': 28*28,
        'output_size': (44, 44),
        'kernel_size': (28, 28),
        'stride': 28,
        'kernel_output_size': (28,28),
        'n': 1,
        'sigma': 2, # 邻近几个神经元（高斯）
        'code_num': 1,
        'lr': 0.3,
    },
    'classifier':{
        'input_size': 44*44,
        'output_size': 10,
    }
}
mpSOM_MNIST_1layers = {
    'save_path': 'model/mpSOM_MNIST_1layers/',
    'layer1':{
        'input_size': 28*28,
        'output_size': (44, 44),
        'kernel_size': (28, 28),
        'stride': 28,
        'kernel_output_size': (28,28),
        'n': 5,
        'sigma': 2, # 邻近几个神经元（高斯）
        'code_num': 20,
        'lr': 0.3,
    },
    'classifier':{
        'input_size': 44*44,
        'output_size': 10,
    }
}
mpSOM_MNIST_1layers_patch = {
    'save_path': 'model/mpSOM_MNIST_1layers_patch/',
    'layer1':{
        'input_size': 14*14,
        'output_size': (44, 44),
        'kernel_size': (14, 14),
        'stride': 14,
        'kernel_output_size': (28,28),
        'n': 5,
        'sigma': 2, # 邻近几个神经元（高斯）
        'code_num': 20,
        'lr': 0.3,
    },
    'classifier':{
        'input_size': 44*44,
        'output_size': 10,
    }
}
mpSOM_MNIST_1layers_patch_overlap = {
    'save_path': 'model/mpSOM_MNIST_1layers_patch_overlap/',
    'layer1':{
        'input_size': 14*14,
        'output_size': (44, 44),
        'kernel_size': (14, 14),
        'stride': 7,
        'kernel_output_size': (28,28),
        'n': 5,
        'sigma': 2, # 邻近几个神经元（高斯）
        'code_num': 20, # 训练时用20个，在show的时候可以减少，以便于显示
        'lr': 0.3,
    },
    'classifier':{
        'input_size': 44*44,
        'output_size': 10,
    }
}

mpSOM_CIFAR_1layers = {
    'save_path': 'model/mpSOM_CIFAR_1layers/',
    'layer1':{
        'input_size': 3*16*16,
        'output_size': (44, 44),
        'kernel_size': (16, 16),
        'stride': 4,
        'kernel_output_size': (32,32),
        'n': 5,
        'sigma': 2, # 邻近几个神经元（高斯）
        'code_num': 100,
        'lr': 0.3,
    },
    'classifier':{
        'input_size': 44*44,
        'output_size': 10,
    }
}
