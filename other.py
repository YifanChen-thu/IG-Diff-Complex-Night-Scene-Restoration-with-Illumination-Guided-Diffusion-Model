import os
# name_list = os.listdir("/home/cyf20/datasets/lol_dataset/our485/low")
# print("namelist[0]:{},[1]:{}".format(name_list[0],name_list[1]))
# print("namelist:{}".format(name_list))
from PIL import Image
import numpy as np
def numpyToPng(numpy_path,png_path):
    npimg = np.load(numpy_path)
    img = Image.fromarray(npimg).convert('RGB')
    img.save(png_path)

root_path = '/home/cyf20/datasets/SDSD/indoor_static_np/GT/pair16_2/'
numpy_paths = os.listdir(root_path)
print(numpy_paths)
png_paths = '/home/cyf20/datasets/SDSD_png/pair16_2/gt/'
os.makedirs(png_paths, exist_ok=True)
for numpy_pth in numpy_paths:
    png_path = png_paths + numpy_pth.split('/')[-1].split('.')[0] + '.png'
    numpy_pth = root_path + numpy_pth
    numpyToPng(numpy_pth,png_path)
