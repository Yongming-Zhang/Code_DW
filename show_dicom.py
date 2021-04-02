import pydicom
import json
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import skimage.io as io

def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    #print(dir(ds))
    print(ds)
    img = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(img)
    io.imshow(img[0], cmap='gray')
    io.show()
    plt.imshow(ds.pixel_array)
    return information

a = loadFileInformation('/mnt/users//20200402/DX/XB175306_20200402_090519_DX/XB175306_20200402_090519_2260_00001_DX.dcm')
#print(a)