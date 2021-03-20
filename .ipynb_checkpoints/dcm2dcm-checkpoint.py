#-*- coding:utf-8 -*-
import SimpleITK as sitk 
import pydicom
from pydicom import dcmread, dicomio 
import os

# path_read:读取原始dicom的文件路径  path_save:保存新的dicom的文件路径
def dcm2dcm(path_read, path_save):
    # GetGDCMSeriesIDs读取序列号相同的dcm文件
    #series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    dcms_source = os.listdir(path_read)
    for id in dcms_source:
        dcms_dir = dicomio.read_file(path_read+id)
        #print(dcms_dir['RescaleSlope'])
        #print(type(dcms_dir['RescaleSlope'].value))
        id = id.split('.')[0]
        if int(id) < 10:
            dcms_target = os.path.join(path_save,'N2D_000'+id+'.dcm')
        elif int(id) < 100 and int(id) >= 10:
            dcms_target = os.path.join(path_save,'N2D_00'+id+'.dcm')
        else:
            dcms_target = os.path.join(path_save,'N2D_0'+id+'.dcm')
        #print('tag1',dcms_dir.dir())
        dcms_dirt = dicomio.read_file(dcms_target)
        dcms_dirt.add_new('RescaleSlope', 'DS', dcms_dir['RescaleSlope'].value)
        dcms_dirt.save_as(dcms_target)
        dcms_dirt = dicomio.read_file(dcms_target)
        print('tag2',dcms_dirt.dir())
        #print(dcms_dirt['RescaleSlope'])
        #print(dcms_dirt['RescaleSlope'].value)
  

if __name__ == '__main__':
    path_dicm = '/mnt/users/drwise_runtime_env/data1/inputdata/1004812/058AD1E9/259A3499/'
    path_save = '/mnt/users/1004812_259A3499_CTA/'
    dcm2dcm(path_dicm, path_save)
    
