#-*-coding:utf-8-*-
import numpy as np
import os
import glob
import SimpleITK as sitk
from skimage import measure
import nibabel as nib
from mitok.utils.mdicom import SERIES
import cv2
from scipy import ndimage
from skimage import morphology
import skimage
import pandas as pd
import json
from skimage.draw import line
from numpy.linalg import norm
import skimage.draw
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def np_to_polydata(np_mat):
    polyData = vtk.vtkPolyData()
    numberOfPoints = len(np_mat)
    points = vtk.vtkPoints()
    for x, y, z in np_mat:
        points.InsertNextPoint(x, y, z)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(numberOfPoints)
    for i in range(numberOfPoints):
        lines.InsertCellPoint(i)
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    return polyData
  
def interp(curve, sample_spacing=1.0):
    curve = np.array(curve)
    polyData = np_to_polydata(curve)
    spline = vtk.vtkSplineFilter()
    cardinal_spine = vtk.vtkCardinalSpline()
    spline.SetInputDataObject(polyData)
    spline.SetSpline(cardinal_spine)
    spline.SetSubdivideToLength()
    spline.SetLength(sample_spacing)
    spline.Update()
    equal_length_pts = vtk_to_numpy(spline.GetOutput().GetPoints().GetData())
    return equal_length_pts

def mark_component(img_arr):
    stats = sitk.LabelShapeStatisticsImageFilter()
    components = sitk.ConnectedComponent(sitk.GetImageFromArray(img_arr.astype(np.uint8)))
    stats.Execute(components)
    label_area = {label:stats.GetNumberOfPixels(label) for label in stats.GetLabels()}
    
    label_area = sorted(label_area.items(), key=lambda label_area:label_area[1], reverse=True)
    #print(label_area)
    label = []
    for i in label_area:
        label.append(i[0])
    #print(label)
    
    components_arr = sitk.GetArrayFromImage(components)
    
    #for idx in range(1, len(label_area)):
    #    label, pixel_num = label_area[idx]
    #    components_arr[components_arr == label] = idx
    
    return label, components_arr

def connected_domain(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)
    '''
    print(num_list_sorted ,output[np.nonzero(output)])
    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask
    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    '''
    print(num_list_sorted, output[np.nonzero(output)])
    return num_list_sorted, output

def connected_component(image):
    # ???????????????3D??????
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return [], image
    # ???????????????region??????
    region = measure.regionprops(label)
    # ????????????????????????????????????
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    # ??????????????????????????????
    '''
    if len(num_list_sorted) > 3:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[3:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        num_list_sorted = num_list_sorted[:3]
    '''
    return num_list_sorted, label
   
def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i

def remove_mix_region(num, volume, out_data):
    after_num = []
    after_volume = []
    for i in range(len(num)):
        if volume[i] > 0:
            after_num.append(num[i])
            after_volume.append(volume[i])
        else:
            out_data[out_data==num[i]] = 0
    return after_num, after_volume, out_data

def save_nii(plaque, data, nii_name):
    dealt_plaque_dir = plaque.split('mask_plaque_round60.nii.gz')[0]
    #dcm_folder = plaque.split('_CTA')[0]
    #series = SERIES(series_path=dcm_folder, strict_check_series=True)
    affine_arr = np.eye(4)
    #affine_arr[0][0] = series.pixel_spacing[0]
    #affine_arr[1][1] = series.pixel_spacing[1]
    #affine_arr[2][2] = series.slice_spacing
    #print(data[np.nonzero(data)])
    #data[np.nonzero(data)] = 1
    data = np.transpose(data, (2, 1, 0))
    data = data.astype('float32')
    plaque_nii = nib.Nifti1Image(data, affine_arr)
    nib.save(plaque_nii, os.path.join(dealt_plaque_dir, nii_name))            
    
def get_brightness(after_num, after_data, img_tensor):
    brightness = []
    for i in after_num:
        brightness.append(round(img_tensor[after_data==i].mean()))
        #print(img_tensor[after_data==i])
    return brightness

def remove_mix_plaque(after_data, after_num, after_volume, brightness, plaque_volumes_mean, plaque_brightness_mean):
    dealt_num = []
    dealt_data = after_data.copy()
    for i in range(len(after_num)):
        if after_volume[i] < plaque_volumes_mean or brightness[i] < plaque_brightness_mean:
            dealt_data[dealt_data==after_num[i]] = 0
            dealt_num.append(after_num[i])
    #print('dealt_data',dealt_data[np.nonzero(dealt_data)])
    #print('after_data',after_data[np.nonzero(after_data)])
    return dealt_data, dealt_num

def gengerate_plaque(after_data, dealt_num, erosion_data):
    for i in range(len(dealt_num)):
        erosion_data[after_data==dealt_num[i]] = 1
    return erosion_data

def compute_center_mass(plaque_data, plaque_num):
    plaque_data_center_mass = plaque_data.copy()
    plaque_data_center_mass[np.nonzero(plaque_data_center_mass)] = 1
    plaque_data_center_mass = ndimage.measurements.center_of_mass(plaque_data_center_mass, plaque_data, plaque_num)
    return plaque_data_center_mass

def generate_vessel_mass(plaque, plaque_center_mass, vessel_data):
    for i in plaque_center_mass:
        plaque_data[(int(i[0]),int(i[1]),int(i[2]))] = 240
        vessel_data[(int(i[0]),int(i[1]),int(i[2]))] = 240
    print(np.unique(vessel_data))
    #save_nii(plaque, vessel_data)

def get_volume(vessel_num, vessel_data):
    vessel_volume = []
    for i in range(len(vessel_num)):
        vessel_volume.append(np_count(vessel_data[np.nonzero(vessel_data)], vessel_num[i]))
    return vessel_volume

def plaque2vessel(plaque, plaque_num, out_plaque_data, vessel_num, vessel_left_data, out_vessel_data, vessel_data, all_points):
    if len(plaque_num) != 0:
        plaque_volume = get_volume(plaque_num, out_plaque_data)
        print('plaque', plaque_num, plaque_volume) 
    if len(vessel_num) != 0:
        vessel_volume = get_volume(vessel_num, out_vessel_data)
        print('vessel', vessel_num, vessel_volume)
    '''
    #?????????????????????????????????????????????????????????????????????????????????????????????????????????100
    vessel_left_arr = vessel_left_data.copy()
    plaque_index = []
    plaque_data_volume = []
    for i in range(len(plaque_num)):
        #????????????????????????????????????????????????
        vessel_left_arr[out_plaque_data==plaque_num[i]] = 0
        vessel_left_arr_again = vessel_left_arr.copy()
        vessel_left2plaque_num, vessel_left2plaque_data = connected_component(vessel_left_arr)
        #?????????????????????????????????????????????
        vessel_left2plaque_volume = get_volume(vessel_left2plaque_num, vessel_left2plaque_data)
        print('i', i, vessel_left2plaque_volume)
        if len(vessel_left2plaque_num) > 1:
            for j in range(1, len(vessel_left2plaque_volume)):
                if vessel_left2plaque_volume[j] > 100:
                    temp = vessel_left2plaque_volume[j]
                    print('j', j)
                    print('temp', temp)
                    #????????????????????????????????????????????????????????????????????????????????????
                    vessel_left_arr_again[np.nonzero(out_plaque_data)] = 0
                    vessel_left2plaque_num_again, vessel_left2plaque_data_again = connected_component(vessel_left_arr_again)
                    #????????????????????????????????????????????????????????????????????????????????????
                    vessel_left2plaque_volume = get_volume(vessel_left2plaque_num_again, vessel_left2plaque_data_again)
                    print('vessel_left2plaque_volume_again', vessel_left2plaque_volume)

                    #for k in range(len(plaque_num)):
                    #    vessel_left_arr_again[out_plaque_data==plaque_num[k]] = 0
                    #    vessel_left2plaque_num_again, vessel_left2plaque_data_again = connected_component(vessel_left_arr_again)
                    #    vessel_left2plaque_volume_again = get_volume(vessel_left2plaque_num_again, vessel_left2plaque_data_again)
                    #    print('vessel_left2plaque_volume_again', vessel_left2plaque_volume_again)
                    #??????????????????????????????????????????????????????????????????????????????????????????????????????
                    if vessel_left2plaque_volume[j] < temp:
                        #???????????????????????????????????????????????????
                        temp = vessel_left2plaque_volume[j]
                        plaque_index.append(i)
                        print('pla_ind', plaque_index)
                        plaque_data_volume.append(temp)
                        print('pla_da_vol', plaque_data_volume)
                    else:
                        plaque_index.append(i)
                        print('pla_ind', plaque_index)
                        plaque_data_volume.append(temp)
                        print('pla_da_vol', plaque_data_volume)
            vessel_left_arr_again = vessel_left_arr.copy()
        #vessel_left_arr = vessel_left_data.copy()
    print('plaque_index', plaque_index)
    print('plaque_data_volume', plaque_data_volume)
    '''
    vessel_new_data = vessel_left_data.copy()
    for i in range(len(plaque_num)):
        #print(vessel_new_data[out_plaque_data==plaque_num[i]])
        vessel_new_data[out_plaque_data==plaque_num[i]] = 2
        vessel_data[out_plaque_data==plaque_num[i]] = 2
        #plaque2vessel_num, out_plaque2vessel_data = connected_component(vessel_new_data)
        #if len(plaque2vessel_num) > len(vessel_num):
    print('vessel_new_data', np.unique(vessel_new_data))
    #save_nii(plaque, vessel_new_data)
    vessel_new_data[vessel_new_data==2] = 0
    vessel_data[vessel_data==2] = 0
    plaque2vessel_num, out_plaque2vessel_data = mark_component(vessel_new_data)
    if len(plaque2vessel_num) != 0:
        plaque2vessel_volume = get_volume(plaque2vessel_num, out_plaque2vessel_data)
    print('plaqueandvessel', plaque2vessel_num, plaque2vessel_volume)

    #????????????????????????
    vessel_max_region = vessel_new_data.copy()
    vessel_max_region[out_plaque2vessel_data!=1] = 0
    #print('vessel_max_region volume', np_count(vessel_max_region, 1))
    #vessel_num??????????????????????????????????????????????????????????????????????????????????????????vessel_num[0]??????????????????????????????????????????????????????
    flag = False
    vessel_dilation_arr = np.zeros(vessel_data.shape)
    vessel_dilation_arr[vessel_data==1] = 1

    if len(plaque_num) != 0:
        if len(plaque2vessel_num) > len(vessel_num) or len(plaque2vessel_num) > 1:
            for i in range(1, len(plaque2vessel_num)):
                if plaque2vessel_volume[i] > 100:
                    #??????????????????????????????
                    #??????????????????????????????
                    dilationed_vessel_data = vessel_new_data.copy()
                    dilationed_vessel_data[out_plaque2vessel_data!=plaque2vessel_num[i]] = 0
                    
                    #??????????????????????????????
                    #for p in range(len(plaque_num)):
                    #    plaque_arr[out_plaque_data==plaque_num[p]] = 10 
                        #?????????????????????????????????????????????????????????????????????3??????0.5???0.00001????????????????????????????????????plaque_num[p]??? 

                    #????????????????????????????????????
                    dilationed_vessel_data = vessel_dilation(dilationed_vessel_data)
                    #??????????????????????????????
                    vessel_max_region_copy = vessel_max_region.copy()
                    vessel_max_region_copy = vessel_dilation(vessel_max_region_copy)
                    #??????????????????????????????
                    #coor_index = []
                    #coord_plaque = []
                    for p in range(len(plaque_num)):
                        plaque_arr = np.zeros(vessel_data.shape)
                        plaque_arr[out_plaque_data==plaque_num[p]] = 10 
                        #??????????????????????????????????????????????????????
                        plaque_dilation_intersection = plaque_arr - dilationed_vessel_data
                        print(np.unique(plaque_dilation_intersection))
                        plaque_max_region_intersection = plaque_arr - vessel_max_region_copy
                        plaque_arr = plaque_arr - dilationed_vessel_data - vessel_max_region_copy
                        print(np.unique(plaque_max_region_intersection))
                        if (plaque_dilation_intersection==9).any() and (plaque_max_region_intersection==9).any():
                            coor_index = p
                            coord_plaque = np.argwhere(plaque_arr==9)
                            break
                    print(coor_index)
                    print(len(coord_plaque))
                    
                    #???????????????????????????????????????
                    print(vessel_data.shape)
                    plaque_arr = np.zeros(vessel_data.shape)
                    plaque_arr[out_plaque_data==plaque_num[coor_index]] = 1
                    plaque_centerline_points = []
                    plaque_centerline_arr = np.zeros(vessel_data.shape)
                    for point in all_points:
                        if plaque_arr[int(round(point[2])),int(round(point[1])),int(round(point[0]))] == 1:
                            if [int(round(point[2])),int(round(point[1])),int(round(point[0]))] not in plaque_centerline_points:
                                plaque_centerline_points.append([int(round(point[2])),int(round(point[1])),int(round(point[0]))])
                            plaque_centerline_arr[int(round(point[2])),int(round(point[1])),int(round(point[0]))] = 1
                    print('plaque_centerline_points', plaque_centerline_points, len(plaque_centerline_points))
                    plaque_centerline_arr_num, plaque_centerline_arr_data = mark_component(plaque_centerline_arr)
                    plaque_centerline_arr_volume = get_volume(plaque_centerline_arr_num, plaque_centerline_arr_data)
                    print('plaque_centerline_arr_num', plaque_centerline_arr_num, plaque_centerline_arr_volume) 
                    #??????????????????????????????
                    #p_index = plaque_data_volume.index(plaque2vessel_volume[i])
                    #print('plaque_index[p_index]', plaque_index[p_index])
                    #coord_plaque = np.argwhere(out_plaque_data==plaque_num[plaque_index[p_index]])
                    #print(len(coord_plaque))
                    '''
                    #???????????????????????????????????????
                    intersection_point_coords = []
                    for index in range(len(coord_plaque)):
                        if dilationed_vessel_data[coord_plaque[index][0]][coord_plaque[index][1]][coord_plaque[index][2]] == 1:
                            vessel_dilation_arr[coord_plaque[index][0]][coord_plaque[index][1]][coord_plaque[index][2]] = 3
                            intersection_point_coords.append(coord_plaque[index])

                    #????????????????????????
                    dilationed_vessel_data = vessel_max_region.copy()
                    #??????????????????????????????
                    vessel_region_data = vessel_new_data.copy()
                    vessel_region_data[out_plaque2vessel_data!=plaque2vessel_num[i]] = 0
                    #??????????????????
                    dilationed_vessel_data = vessel_dilation(dilationed_vessel_data)
                    print(len(coord_plaque))
                    for index in range(len(coord_plaque)):
                        if dilationed_vessel_data[coord_plaque[index][0]][coord_plaque[index][1]][coord_plaque[index][2]] == 1:
                            vessel_dilation_arr[coord_plaque[index][0]][coord_plaque[index][1]][coord_plaque[index][2]] = 3
                            intersection_point_coords.append(coord_plaque[index])
                    #print(vessel_dilation_arr.shape)
                    '''

                    for index in range(len(coord_plaque)):
                        vessel_dilation_arr[coord_plaque[index][0]][coord_plaque[index][1]][coord_plaque[index][2]] = 3
                    
                    save_nii(plaque, vessel_dilation_arr, 'mask_pipeline_vessel_plaque_middlepoint_view.nii.gz') 

                    #??????????????????????????????
                    dilationed_vessel_data = vessel_new_data.copy()
                    dilationed_vessel_data[out_plaque2vessel_data!=plaque2vessel_num[i]] = 0
                    #print(dilationed_vessel_data.shape)
                    #print('dilationed_vessel_data volume', np_count(dilationed_vessel_data, 1))
                    print(np.unique(vessel_max_region[dilationed_vessel_data==1]))
                    while vessel_max_region[dilationed_vessel_data==1].any() != 1:
                        #????????????????????????????????????????????????????????????????????????????????????
                        dilationed_vessel_data = vessel_dilation(dilationed_vessel_data)
                    #????????????????????????????????????????????????????????????1????????????????????????????????????
                    #print(vessel_max_region[dilationed_vessel_data==1])
                    first_point_coord = find_coord(vessel_max_region, dilationed_vessel_data)
                    vessel_data[first_point_coord[0]][first_point_coord[1]][first_point_coord[2]] = 2
                    #print(vessel_max_region[dilationed_vessel_data==1].any() == 1)

                    #????????????????????????
                    dilationed_vessel_data = vessel_max_region.copy()
                    #??????????????????????????????
                    vessel_region_data = vessel_new_data.copy()
                    vessel_region_data[out_plaque2vessel_data!=plaque2vessel_num[i]] = 0
                    #print(dilationed_vessel_data.shape)
                    #print('dilationed_vessel_data volume', np_count(dilationed_vessel_data, 1))
                    print(np.unique(vessel_region_data[dilationed_vessel_data==1]))
                    while vessel_region_data[dilationed_vessel_data==1].any() != 1:
                        #???????????????????????????????????????
                        dilationed_vessel_data = vessel_dilation(dilationed_vessel_data)
                    #????????????????????????????????????????????????????????????1????????????????????????????????????
                    #print(vessel_max_region[dilationed_vessel_data==1])
                    second_point_coord = find_coord(vessel_region_data, dilationed_vessel_data)
                    vessel_data[second_point_coord[0]][second_point_coord[1]][second_point_coord[2]] = 2
                    #generate_point_lines(z_min, z_max, y_min, y_max, x_min, x_max, vessel_data)
                    
                    vessel_data, middle_point = generate_point_lines(first_point_coord[0], first_point_coord[1], first_point_coord[2], second_point_coord[0], second_point_coord[1], second_point_coord[2], vessel_data)
                    #vessel_data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 2

                    vessel_data_copy = vessel_data.copy()
                    vessel_data_copy[out_plaque2vessel_data!=plaque2vessel_num[i]] = 0
                    #?????????????????????????????????
                    #intersection_point_coords = find_intersection_coords(vessel_data_copy, vessel_data.copy())
                    #??????????????????????????????????????????????????????????????????????????????
                    for j in range(len(coord_plaque)):
                        generate_pipeline(coord_plaque[j], middle_point, vessel_data)

                    print(vessel_data[vessel_data==2])
                    
                    flag = True
                    #vessel_new_data = generate_line(first_point_coord, second_point_coord, vessel_new_data)
                    #rr, cc = line(first_point_coord[1], first_point_coord[2], second_point_coord[1], second_point_coord[2])
                    #vessel_new_data[rr, cc] = 2
            if flag:

                #??????=1?????????=2
                #?????????????????????????????????
                out_plaque_data[vessel_data==2] = 0
                #???????????????????????????1
                vessel_data[vessel_data==2] = 1
                #????????????2
                vessel_data[np.nonzero(out_plaque_data)] = 2

                save_nii(plaque, vessel_data, 'mask_pipeline_vessel_plaque_middlepoint.nii.gz')
                vessel_data[vessel_data==2] = 1
                data_num, out_data = mark_component(vessel_data)
                print('repaired_vessel_data', data_num, get_volume(data_num, out_data))
            else:
                print('??????????????????100???????????????????????????')
        else:
            print('???????????????????????????????????????')
    else:
        print('????????????????????????')

def generate_point_lines(fz, fy, fx, sz, sy, sx, vessel_data):
    point_line_length = math.ceil(math.sqrt(abs(fz-sz)**2+abs(fy-sy)**2+abs(fx-sx)**2))
    print('point_line_length', point_line_length)
    middle_point_length = round(point_line_length/2)
    z_dist = abs(fz-sz)
    y_dist = abs(fy-sy)
    x_dist = abs(fx-sx)
    points = []
    middle_point = 0
    for i in range(0, point_line_length+1):
        point = []
        if z_dist != 0:
            if fz-sz < 0:
                point.append(fz + int(round(i/(point_line_length/z_dist))))
                #point.append(fz + int((i//(point_line_length/z_dist))))
            elif fz-sz > 0:
                point.append(fz - int(round(i/(point_line_length/z_dist))))
                #point.append(fz - int((i//(point_line_length/z_dist))))
        else:
            point.append(fz)
        if y_dist != 0:
            if fy-sy < 0:
                point.append(fy + int(round(i/(point_line_length/y_dist))))
                #point.append(fy + int((i//(point_line_length/y_dist))))
            elif fy-sy > 0:
                point.append(fy - int(round(i/(point_line_length/y_dist))))
                #point.append(fy - int((i//(point_line_length/y_dist))))
        else:
            point.append(fy)
        if x_dist != 0:
            if fx-sx < 0:
                point.append(fx + int(round(i/(point_line_length/x_dist))))
                #point.append(fx + int((i//(point_line_length/x_dist))))
            elif fx-sx > 0:
                point.append(fx - int(round(i/(point_line_length/x_dist))))
                #point.append(fx - int((i//(point_line_length/x_dist))))
        else:
            point.append(fx)
        points.append(point)
        #??????????????????
        if i == middle_point_length:
            middle_point = point
    print('points', points)
    #final_points = []
    #??????????????????????????????????????????????????????????????????
    for i in range(1, len(points)):
        if points[i][0] != points[i-1][0] and points[i][1] != points[i-1][1] and points[i][2] != points[i-1][2]:
            #final_points.append([points[i-1][0], points[i][1], points[i][2]])
            points.insert(i, [points[i-1][0], points[i][1], points[i][2]])
    #final_points.extend(points)
    print('final_points', points)
    print('middle_point', middle_point)
    
    #????????????????????????????????????
    vessel_middle_point = np.zeros(vessel_data.shape)
    vessel_middle_point[middle_point[0]][middle_point[1]][middle_point[2]] = 1
    vessel_middle_point_dilation = vessel_dilation(vessel_middle_point)
    #vessel_middle_point_dilation = vessel_dilation(vessel_middle_point_dilation)
    
    for data in points:
        vessel_data[data[0]][data[1]][data[2]] = 2

    vessel_data[np.nonzero(vessel_middle_point_dilation)] = 2

    return vessel_data, middle_point

def generate_pipeline(intersection_point_coords, middle_point, vessel_data):
    fz, fy, fx = intersection_point_coords[0], intersection_point_coords[1], intersection_point_coords[2]
    sz, sy, sx = middle_point[0], middle_point[1], middle_point[2]
    point_line_length = math.ceil(math.sqrt(abs(fz-sz)**2+abs(fy-sy)**2+abs(fx-sx)**2))
    #print('point_line_length', point_line_length)
    z_dist = abs(fz-sz)
    y_dist = abs(fy-sy)
    x_dist = abs(fx-sx)
    points = []
    for i in range(0, point_line_length+1):
        point = []
        if z_dist != 0:
            if fz-sz < 0:
                point.append(fz + int(round(i/(point_line_length/z_dist))))
                #point.append(fz + int((i//(point_line_length/z_dist))))
            elif fz-sz > 0:
                point.append(fz - int(round(i/(point_line_length/z_dist))))
                #point.append(fz - int((i//(point_line_length/z_dist))))
        else:
            point.append(fz)
        if y_dist != 0:
            if fy-sy < 0:
                point.append(fy + int(round(i/(point_line_length/y_dist))))
                #point.append(fy + int((i//(point_line_length/y_dist))))
            elif fy-sy > 0:
                point.append(fy - int(round(i/(point_line_length/y_dist))))
                #point.append(fy - int((i//(point_line_length/y_dist))))
        else:
            point.append(fy)
        if x_dist != 0:
            if fx-sx < 0:
                point.append(fx + int(round(i/(point_line_length/x_dist))))
                #point.append(fx + int((i//(point_line_length/x_dist))))
            elif fx-sx > 0:
                point.append(fx - int(round(i/(point_line_length/x_dist))))
                #point.append(fx - int((i//(point_line_length/x_dist))))
        else:
            point.append(fx)
        points.append(point)
    #print('points', points)
    #??????????????????????????????????????????????????????????????????
    for i in range(1, len(points)):
        if points[i][0] != points[i-1][0] and points[i][1] != points[i-1][1] and points[i][2] != points[i-1][2]:
            #final_points.append([points[i-1][0], points[i][1], points[i][2]])
            points.insert(i, [points[i-1][0], points[i][1], points[i][2]])
    #print('final_points', points)
    for data in points:
        vessel_data[data[0]][data[1]][data[2]] = 2
    return vessel_data

def generate_line(first_point_coord, second_point_coord, vessel_new_data): 
    c = lambda *x: np.array(x, dtype=float)
    vector_angle = lambda V, U: np.arccos(norm(np.dot(V, U)) / (norm(V) + norm(U)))

    r = 1 # radius of cylinder
    C0 = c(first_point_coord[2], first_point_coord[1], first_point_coord[0]) # first (x,y,z) point of cylinder
    C1 = c(second_point_coord[2], second_point_coord[1], second_point_coord[0]) # second (x,y,z) point of cylinder

    C = C1 - C0

    X, Y, Z = np.eye(3)

    theta = vector_angle(Z, C)
    print('theta={} deg'.format(theta / np.pi * 180))

    minor_axis = r
    major_axis = r / np.cos(theta)
    print('major_axis', major_axis)

    alpha = vector_angle(X, C0 + C)
    print('alpha={} deg'.format(alpha / np.pi * 180))

    #data = np.zeros([100, 100, 100])
    nz, ny, nx = vessel_new_data.shape
    nz1 = min(second_point_coord[0], first_point_coord[0])
    nz2 = max(second_point_coord[0], first_point_coord[0])
    for z in range(nz1, nz2):
        lam = - (C0[2] - z)/C[2]
        P = C0 + C * lam
        y, x = skimage.draw.ellipse(P[1], P[0], major_axis, minor_axis, shape=(ny, nx), rotation=alpha)
        vessel_new_data[z, y, x] = 2
    return vessel_new_data

def find_coord(vessel_max_region, dilationed_vessel_data):
    coord_index = np.argwhere(dilationed_vessel_data==1)
    #print(coord_index.shape)
    for index in range(coord_index.shape[0]):
        if vessel_max_region[coord_index[index][0]][coord_index[index][1]][coord_index[index][2]] == 1:
            point_coord = coord_index[index]
            print(coord_index[index])
    '''
    z, x, y = dilationed_vessel_data.shape
    coords = []
    for i in range(z):
        for j in range(x):
            for k in range(y):
                if dilationed_vessel_data[i][j][k] == n:
                    coords.append((i,j,k))
    for m in coords:
        if vessel_max_region[m] == 1:
            print(m)
    '''
    return point_coord

def find_intersection_coords(intersection_data, vessel_data):
    print(vessel_data.shape, intersection_data.shape)
    #print(vessel_data[np.nonzero(vessel_data[intersection_data])])
    coord_index = np.argwhere(vessel_data[intersection_data]==1)
    print(coord_index)
    point_coord = []
    for index in range(coord_index.shape[0]):
        #if vessel_data[coord_index[index][0]][coord_index[index][1]][coord_index[index][2]] == 1:
        point_coord.append(coord_index[index])
    print('point_coord', point_coord)
    return point_coord

def plaque_erode(dealt_data):
    dealt_data[np.nonzero(dealt_data)] = 1
    print(dealt_data[np.nonzero(dealt_data)])

    struct = ndimage.generate_binary_structure(3, 1)
    #kernel = skimage.morphology.ball(1, dtype=np.int)
    #dealt_data = ndimage.binary_dilation(dealt_data, struct, 1).astype(dealt_data.dtype)
    # Scipy erosion
    erosion_data = ndimage.binary_erosion(dealt_data).astype(dealt_data.dtype)
    #erosion_data = morphology.erosion(dealt_data, morphology.ball(3, dtype=np.int))
    # Scikit-image erosion
    #erosion_data = morphology.binary_erosion(dealt_data.astype(uint), struct)

    #???????????????5*5
    #kernel = np.ones((5,5), np.uint8)
    #dealt_data = cv2.cvtColor(dealt_data, cv2.COLOR_BGR2GRAY)
    #????????????????????????????????????
    #erosion_data = cv2.erode(dealt_data, kernel)
    #???????????????
    #dilate_data = cv2.dilate(erosion, kernel)
    return erosion_data

def vessel_dilation(dealt_data):
    dilation_data = ndimage.binary_dilation(dealt_data).astype(dealt_data.dtype)
    return dilation_data

def remove_other_vessel(vessel_data, out_vessel_data):
    vessel_left_data = vessel_data.copy()
    vessel_left_data[out_vessel_data!=1] = 0
    return vessel_left_data

#plaque_dir = '/mnt/users/ffr_plaque_mask/'
plaque_dir = '/mnt/users/ffr_datasets/ffr_cpr_mask_newmap/'
vessel_dir = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' 
#plaque_list = glob.glob(os.path.join(plaque_dir, '*', '*', 'mask_plaque.nii.gz'))
plaque_list = glob.glob(os.path.join(plaque_dir, '*', '*', 'mask_plaque_round60.nii.gz'))
#vessel_list = glob.glob(os.path.join(vessel_dir, plaque.split('/')[4], '*', plaque.split('/')[5]+'_CTA', 'mask_source/mask_vessel.nii.gz'))
counts = 0
without_plaque_list = []
all_plaque_volumes = []
all_plaque_brightness = []
plaques = []
#plaque_datasets = [1036619, 1036625, 1036603, 1036612]
#plaque_datasets = [1073318]
#broken_vessels = [1073332, 1073332, 1036627, 1036604, 1073308, 1036623, 1073309, 1036609, 1073297, 1073318, 1073318, 1036617, 1036617, 1073300, 1073298, 1022836]
broken_vessels = ['1036604_60_0416']#['1036604_60_0416', '1036623', '1073309', '1073332', '1073330', '1073318', '1036617']#['1036623', '1073309', '1073332', '1073330', '1073318', '1036617'] #[1036604_60_0416]
for plaque in plaque_list:
    #print(plaque.split('/')[4])
    if plaque.split('/')[5] in broken_vessels: #int(plaque.split('/')[4]) > 0: 
        #continue
        #plaque = '/mnt/users/ffr_plaque_mask/1073318/AF7B89E9/mask_plaque.nii.gz'
        print(plaque)
        plaques.append(plaque)
        plaque_data = sitk.ReadImage(plaque)
        plaque_data = sitk.GetArrayFromImage(plaque_data)
        vessel_list = glob.glob(os.path.join(vessel_dir, plaque.split('/')[5], '*', plaque.split('/')[6], 'mask_source/mask_vessel.nii.gz'))[0]
        vessel_data = sitk.ReadImage(vessel_list)
        vessel_data = sitk.GetArrayFromImage(vessel_data)   
        centerline_list = glob.glob(os.path.join(vessel_dir, plaque.split('/')[5], '*', plaque.split('/')[6], 'result.json'))[0]
        
        all_points = []
        with open(centerline_list, 'r') as f:
            datas = json.load(f)
            centerlines = datas["center_lines"]
            for point in centerlines:
                all_points.extend(point["points"])
        all_points = interp(all_points, sample_spacing=0.5)
        #print(all_points, len(all_points))
        #print(vessel_data[np.nonzero(vessel_data)])
        #num, out_data = connected_domain(data)
        plaque_num, out_plaque_data = mark_component(plaque_data)
        #plaque_center_mass = compute_center_mass(out_plaque_data, plaque_num)
        #for i in plaque_center_mass:
        #    print(plaque_data[(int(i[0]),int(i[1]),int(i[2]))],(int(i[0]),int(i[1]),int(i[2])))
        #print(plaque_center_mass)
        #print(plaque_num, out_plaque_data[np.nonzero(out_plaque_data)])
        vessel_num, out_vessel_data = mark_component(vessel_data)
        #print(vessel_num, out_vessel_data[np.nonzero(out_vessel_data)])
        vessel_left_data = remove_other_vessel(vessel_data, out_vessel_data)
        plaque2vessel(plaque, plaque_num, out_plaque_data, vessel_num[:1], vessel_left_data, out_vessel_data[out_vessel_data==1], vessel_data, all_points)
        #generate_vessel_mass(plaque, plaque_center_mass, vessel_data)
print(len(plaques))

