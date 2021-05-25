#-*-coding:utf-8-*-
import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageDraw 

#因为工程化生成的cpr斑块mask仅是在cpr上的斑块轮廓，需要将斑块轮廓转化为斑块mask，所有这里要先提取出cpr并生成只有cpr斑块的轮廓，之后对cpr斑块轮廓生成斑块mask。
tar_dir = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata/' #'/mnt/users/ffr_10datasets/ffr_cpr_mask/'#diagnose_debug/segmentation/'
sou_dir = '/mnt/users/ffr_datasets/ffr_cpr_mask_dealt/' #'/data1/zhangyongming/cprmask'
contours_dir = '/mnt/users/ffr_datasets/ffr_cpr_mask_contours/'
pat_list = os.listdir(tar_dir)
name_list = ['1036623', '1073309', '1073332', '1073330', '1073318', '1036617']#['1036604_20_0416', '1036604_60_0416']#['1036623', '1073309', '1073332', '1073330', '1073318', '1036617'] #['1036631', '1036632', '1036610', '1036619', '1036630', '1036615', '1036625', '1073332', '1036607', '1036624']
for pat in pat_list:
    if pat not in name_list:
        continue 
    tar_list = os.listdir(glob.glob(os.path.join(tar_dir+pat, '*'))[0])
    #print(tar_list)
    for tar in tar_list:
        if 'CTA' not in tar:
            continue 
        images_list = os.listdir(glob.glob(os.path.join(tar_dir+pat, '*', tar, 'diagnose_debug/segmentation'))[0])
        #print(images_list)
        tar_pat_dir = glob.glob(os.path.join(tar_dir+pat, '*', tar, 'diagnose_debug/segmentation'))[0]
        for image in images_list:
            #tar_dir = glob.glob(os.path.join(tar_dir+pat, '*', tar, 'diagnose_debug/segmentation'))[0]
            im = Image.open(os.path.join(tar_pat_dir, image))
            pix = im.load()
            w, h = im.size[0], im.size[1]
            coord_xy = []
            #img = np.zeros((w, h), dtype="uint8")
            for x in range(w):
                for y in range(h):
                    r, g, b = pix[x, y]
                    if r == 255 and g == 0 and b == 0:
                        pix[x, y] = 255, 255, 255
                    else:
                        pix[x, y] = 0, 0, 0

            if not os.path.exists(contours_dir+pat+'/'+tar):
                os.makedirs(os.path.join(contours_dir+'/'+pat+'/'+tar))
            print(os.path.join(contours_dir, pat, tar, image))
            im.save(contours_dir+pat+'/'+tar+'/'+image)

            contours_img_dir = contours_dir+pat+'/'+tar+'/'+image
            contours_img = cv2.imread(contours_img_dir)
            mask = cv2.imread(contours_img_dir, 0)
            contours_img_gray = cv2.cvtColor(contours_img, cv2.COLOR_BGR2GRAY)
            ret, binaryzation = cv2.threshold(contours_img_gray, 0, 255, 0)
            contours, hierarchy = cv2.findContours(binaryzation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #print(contours)
            for i in range(len(contours)):
                cv2.drawContours(mask, contours, i, 255, -1)

            if not os.path.exists(sou_dir+pat+'/'+tar):
       	       os.makedirs(os.path.join(sou_dir+'/'+pat+'/'+tar))
            cv2.imwrite(sou_dir+pat+'/'+tar+'/'+image, mask)

       