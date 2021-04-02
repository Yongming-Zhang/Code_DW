import os
import shutil

images_label = '/mnt/data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
images_path = '/mnt/data/rsna-pneumonia-detection-challenge/stage_2_train_images_jpg/'
new_images_path = '/mnt/data/rsna-pneumonia-detection-challenge/new_stage_2_train_images_jpg/'

with open(images_label) as f:
    samples = [x.strip().split(',') for x in f.readlines()]
samples = samples[1:]
    
data_filename = []
data_sets = []
for data in samples:
    if data[0] not in data_filename:
        data_filename.append(data[0])
        data_sets.append(data)
samples = data_sets

data_infos = []
i = 0
j = 0
for filename, _, _, _, _, gt_label in samples: 
    if int(gt_label) == 0 and i < 6993:#6993
        #print(filename,gt_label)
        path = os.path.join(new_images_path, 'NORMAL/')
        if not os.path.exists(path):
            os.makedirs(path)
        img_path = os.path.join(images_path, filename+'.jpg')
        shutil.copyfile(img_path, os.path.join(path, filename+'.jpg'))
        i += 1
    elif int(gt_label) == 1 and j < 4659:#4659
        #print(filename,gt_label)
        path = os.path.join(new_images_path, 'PNEUMONIA/')
        if not os.path.exists(path):
            os.makedirs(path)
        img_path = os.path.join(images_path, filename+'.jpg')
        shutil.copyfile(img_path, os.path.join(path, filename+'.jpg'))
        j += 1

print('finished!')