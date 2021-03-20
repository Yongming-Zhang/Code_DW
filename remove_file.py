import os

file_dir = '/mnt/users/code/mmclassification/work_dirs/chest_pediatric_loadimagenet_batch64x1_ft_eval_20201217'
files = os.listdir(file_dir)
for name in files:
    if name.split('_')[0] == 'epoch':
        na = name.split('_')[1]
        na = na.split('.')[0]
        if int(na) < 100:
            print(name)
            os.remove(os.path.join(file_dir+'/'+name))
