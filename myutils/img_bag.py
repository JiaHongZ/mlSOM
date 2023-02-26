import PIL.Image as Image
import matplotlib.pyplot as plt
import os
import pickle

path = r'E:\image_denoising\zzz-finished\DRNet\DRNet\DRNet\trainsets\train5544'
image_list = os.listdir(path)[:1000]
saveimge = []

for name in image_list:
    img_path = os.path.join(path,name)
    img = Image.open(img_path)
    img = img
    # plt.imshow(img)
    # plt.show()
    saveimge.append(img)

with open('imgbag.txt','wb') as file:
    pickle.dump(saveimge, file)

# if __name__ == '__main__':
#     with open('imgbag.txt', 'rb') as file:
#         imgs = pickle.load(file)
#     img = imgs[10]
#     img = img.resize((224,224),Image.ANTIALIAS)
#     plt.imshow(img)
#     plt.show()