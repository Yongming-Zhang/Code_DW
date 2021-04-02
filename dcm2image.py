import pydicom
import cv2
import os
from tqdm import tqdm
 
path = '/mnt/data/rsna-pneumonia-detection-challenge/stage_2_train_images/'
out_path = '/mnt/data/rsna-pneumonia-detection-challenge/stage_2_train_images_jpg/'

path_list = os.listdir(path)
for pa in tqdm(path_list):
    ds = pydicom.read_file(path+pa)  #读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imwrite(out_path+pa.split('.')[0]+'.jpg',img)
