# mlSOM
Extracting the Brain-like Representation by an Improved Self-Organizing Map for Image Classification

This paper has been accepted by ICASSP 2023.

Backpropagation-based supervised learning has achieved great success in computer vision tasks. However, its biological plausibility is always controversial. Recently, the bio-inspired Hebbian learning rule (HLR) has received extensive attention. Self-Organizing Map (SOM) uses the competitive HLR to establish connections between neurons, obtaining visual features in an unsupervised way. Although the representation of SOM neurons shows some brain-like characteristics, it is still quite different from the neuron representation in the human visual cortex. This paper proposes an improved SOM with multi-winner, multi-code, and local receptive field, named mlSOM. We observe that the neuron representation of mlSOM is similar to the human visual cortex. Furthermore, mlSOM shows a sparse distributed representation of objects, which has also been found in the human inferior temporal area. In addition, experiments show that mlSOM achieves better classification accuracy than the original SOM and other state-of-the-art HLR-based methods. The code of DRNet is accessible at https://github.com/JiaHongZ/mlSOM.

2024.6.5：
We found that this code does not reproduce the results on the latest version of PyTorch. The known working environment is Python 3.6 and PyTorch 1.6.0.
