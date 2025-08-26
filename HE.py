import cv2 as cv
img = cv.imread("/home/cyf20/datasets/lol_dataset/eval15/low/493.png",0)

import sys
import os

import cv2
import json

import retinex

data_path = 'data'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print( 'Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

    
img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
cv2.imshow('MSRCR retinex', img_msrcr)
cv2.imwrite("MSRCR_retinex.tif",img_msrcr)

# for img_name in img_list:
#     if img_name == '.gitkeep':
#         continue
    
#     img = cv2.imread(os.path.join(data_path, img_name))

#     print('msrcr processing......')
#     img_msrcr = retinex.MSRCR(
#         img,
#         config['sigma_list'],
#         config['G'],
#         config['b'],
#         config['alpha'],
#         config['beta'],
#         config['low_clip'],
#         config['high_clip']
#     )
#     cv2.imshow('MSRCR retinex', img_msrcr)
#     cv2.imwrite("MSRCR_retinex.tif",img_msrcr);


#     print('amsrcr processing......')
#     img_amsrcr = retinex.automatedMSRCR(
#         img,
#         config['sigma_list']
#     )
#     cv2.imshow('autoMSRCR retinex', img_amsrcr)
#     cv2.imwrite('AutomatedMSRCR_retinex.tif', img_amsrcr)


#     print('msrcp processing......')
#     img_msrcp = retinex.MSRCP(
#         img,
#         config['sigma_list'],
#         config['low_clip'],
#         config['high_clip']        
#     )    

#     shape = img.shape
#     cv2.imshow('Image', img)

#     cv2.imshow('MSRCP', img_msrcp)
#     cv2.imwrite('MSRCP.tif', img_msrcp)
#     cv2.waitKey()







# img = cv.resize(img, None, fx=0.5, fy=0.5)
# # 创建CLAHE对象
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # 限制对比度的自适应阈值均衡化
# dst = clahe.apply(img)  #限制对比度自适应直方图均衡化(CLAHE)
# # 使用全局直方图均衡化
# equa = cv.equalizeHist(img) #全局直方图均衡化(HE)
# # 分别b原图，CLAHE，HE
# cv.imwrite('./493_clahe.png',dst)
# cv.imwrite('./493_equa.png',equa)

# img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
# img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
# img_output = cv.cvtColor(img_yuv,cv.COLOR_YUV2BGR)

# img_yuv2 = cv.cvtColor(img,cv.COLOR_BGR2YUV)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img_yuv2[:,:,0] =  clahe.apply(img_yuv2)
# img_output2 = cv.cvtColor(img_yuv2,cv.COLOR_YUV2BGR)

# cv.imwrite('./493_equa2.png',img_output)
# cv.imwrite('./493_clahe2.png',img_output2)
