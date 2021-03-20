import cv2
import torch
import torch.nn as nn
import numpy as np
from skimage import measure
import torch.nn.functional as F

from mitok.image.cv_gpu import label, fill_holes, tensor_slice_bool_assign
from mitok.utils.decrypt_weights import LoadTorchWeights
from mitok.utils.compatible import rebuild_tensor_v2
from .cctav_model.vessel_segmentation.resunet import DASEResNet18rad
from .cctav_model.vessel_segmentation.vt_net import DASEResNetW
from .cctav_model.vessel_segmentation.completion_net import DASEResLstmNet18
from .cctav_model.vessel_segmentation.multiresunet import DAMultiResUnet3_4

from functools import reduce
from .cctav_utils.patch import get_patch_coords
from .cctav_utils.data import Volume3DLoader, Volume3DLoaderCasecade, set_window_wl_ww

# check_pytorch_compatible()

model_dict = {'da_seresnet18': DASEResNet18rad}

def tensor_resize(image, shape, is_label=True):
    x = torch.unsqueeze(image, dim=0)
    x = torch.unsqueeze(x, dim=0)
    if is_label:
        y = F.interpolate(x.float(), shape, mode='nearest')
    else:
        y = F.interpolate(x.float(), shape, mode= 'trilinear', align_corners=True)
    y = y.to(image.dtype)[0, 0]
    return y

class VesselSegmentRadius(object):
    
    def __init__(self, gpu_id, logger, conf, crop=False):
        self.logger = logger
        self.device = torch.device("cuda:"+str(gpu_id) if gpu_id>=0 else "cpu")
        self.cpu_device = torch.device("cpu")
        # torch.cuda.set_device(int(gpu_id))
        self.heatmap = conf['B_heatmap']
        self.num_class = conf['num_class']
        self.crop = crop
        self.mode = conf['mode']
        self.use_completion = conf['B_use_completion']
        self.use_completion_post = conf['B_use_completion_post']
        self.patch_adptive = conf['patch_adptive']
        self.use_coord = conf['use_coord']
        
        self.net = DASEResNet18rad(self.num_class, k=24, heatmap=True)
        #self.net = DAMultiResUnet3_4(2, k=32)
        ct = LoadTorchWeights(conf['weights_radius'])
        self.net.load_state_dict(ct['model'])
        # self.net = nn.DataParallel(self.net, device_ids=[int(gpu_id)])
        self.net.eval()
        '''
        self.net_hp = DASEResNet18rad(self.num_class, k=24, heatmap=True)
        ct_hp = LoadTorchWeights(conf['weights_3d'])
        self.net_hp.load_state_dict(ct_hp['model'])
        self.net_hp.eval()
        '''
        self.batch_size = conf['batch_size']
        self.patch_size = conf['patch_size']
        
    def seg_3d(self, imt):
        print('1,seg_3d')
        patch_size = self.patch_size
        volume_size = imt.shape
        
        coords = get_patch_coords(patch_size, volume_size)
        
        p_x, p_y, p_z = patch_size
        v_x, v_y, v_z = volume_size
        
        test_set = Volume3DLoader(imt, coords, patch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        seg = torch.FloatTensor(self.num_class, v_x, v_y, v_z).zero_().to(self.device)
        #seg = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
        if self.heatmap:
            hp = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
        
        self.net.to(self.device)
        for i, (image, coord) in enumerate(test_loader):

            image = image.to(torch.float32)
            out = self.net(image)

            pred = F.softmax(out['r'], dim=1)
            for idx in range(image.size(0)):
                sx, ex = coord[idx][0][0], coord[idx][0][1]
                sy, ey = coord[idx][1][0], coord[idx][1][1]
                sz, ez = coord[idx][2][0], coord[idx][2][1]

                seg[:, sx:ex, sy:ey, sz:ez] = torch.max(seg[:, sx:ex, sy:ey, sz:ez], pred[idx])
                if self.heatmap:
                    hp[sx:ex, sy:ey, sz:ez] += out['hp'][idx][0]
                    num[sx:ex, sy:ey, sz:ez] += 1
                
        self.net.to(self.cpu_device)
        
        seg, _ = seg[1:, ...].max(0)
        prob = seg.float()
        hp = (hp / num.float()).cpu().numpy() if self.heatmap else np.zeros_like(prob)
        return prob, hp
    
    def resume_deleted_vessel(self, seg, z, cslice, cimage, threshold=1000):
        """
        :param seg: (x, y, z) order
                   z: remove_extra_vessel函数里删除的主动脉弓的层面
                   cslice, cimage: 删除的主动脉弓层面的血管mask的坐标
        """
        print('2,resume_deleted_vessel')
        if  z+1 >= seg.shape[-1] or z == 0 or cslice is None:
            return seg
        
        _, props = label(seg[:,:,z-1], device=self.device, to_numpy=False,
                         connectivity=2, calc_prop=False, min_volumn=threshold,
                         top_k_region=1)
        if len(props) == 0:
            return seg
        
        _, props = label(seg[:,:,z+1], device=self.device, to_numpy=False,
                         connectivity=2, calc_prop=False, min_volumn=threshold,
                         top_k_region=1)
        if len(props) == 0:
            return seg
        
        tensor_slice_bool_assign(seg[:,:,z], cslice, cimage, 1)
        return seg
        
    
    def remove_extra_vessel(self, seg, threshold=1000):
        """
        :param seg: (x, y, z) order
        标注的时候主动脉未全标，导致分割的时候，最上面几层的主动脉分割比较零碎，去除这些分割
        """        
        print('3,remove_extra_vessel')
        z = 0
        area = 0
        while z < seg.shape[-1]:
            if seg[:, :, z].sum() == 0:
                z = z + 1
                continue
            _, props = label(seg[:,:,z], device=self.device, to_numpy=False,
                             connectivity=2, calc_prop=True,
                             min_volumn=threshold, top_k_region=1)
            if len(props) > 0:
                max_prop = props[0]
                area = max_prop.area
                tensor_slice_bool_assign(seg[:,:,z], max_prop.slice,
                                         max_prop.image, 0)
                break
            z = z + 1
        
        #z = z + 1
        #while z < seg.shape[-1]:
        #    _, props = label(seg[:,:,z], device=self.device, to_numpy=False,
        #                     connectivity=2, calc_prop=True,
        #                     min_volumn=threshold, top_k_region=1)
        #    if len(props) <= 0:
        #        break
        #    max_prop = props[0]
        #    if area / max_prop.area >= 0.8:
        #        tensor_slice_bool_assign(seg[:,:,z], max_prop.slice,
        #                                 max_prop.image, 0)
        #        break
        #    else:
        #        area = max_prop.area
        #        z = z + 1
        
        start = z
        z = min(seg.shape[2], z + 1)
        res = []
        while z < seg.shape[-1]:
            _, props = label(seg[:,:,z], device=self.device, to_numpy=False,
                             connectivity=2, calc_prop=True,
                             min_volumn=threshold, top_k_region=1)
            if len(props) == 0:
                z = z + 1
                res.append(0)
                continue
            max_prop = props[0]
            
            im = np.zeros((seg.shape[0], seg.shape[1]), 'uint8')
            im[max_prop.slice][max_prop.image.cpu().numpy().astype(bool)] = 1
            contors, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ee = cv2.fitEllipse(contors[0])
            mm = np.zeros((seg.shape[0], seg.shape[1]), 'uint8')
            mm = cv2.ellipse(mm, ee, 1, cv2.FILLED)
            dd = (mm * im).sum() / (mm.sum()+im.sum())
            res.append(dd)
            
            if len(res) < 3:
                z = z + 1
                continue
            a, b, c = res[-3:]
            #print(z, a, b, c)
            t1 = min(a, min(b, c)) >= 0.45
            t2 = abs(a - b) < 0.02 and abs(c - b) < 0.02
            t = t1 and t2
            if not t:
                z = z + 1
                continue
            
            _, props = label(seg[:,:,min(z+5, seg.shape[2]-1)],
                             device=self.device, to_numpy=False,
                             connectivity=2, calc_prop=False,
                             top_k_region=1)
            if len(props) == 0:
                m_a = 0
            else:
                m_p = props[0]
                m_a = m_p.area
            
            if abs(m_a - max_prop.area) <= 100:
                break
            
            z = z + 1
        
        cslice = cimage = None
        if z - start <= 15:
            _, props = label(seg[:,:,z-1], device=self.device, to_numpy=False,
                             connectivity=2, calc_prop=True,
                             top_k_region=1)
            if len(props) > 0:
                max_prop = props[0]
                tensor_slice_bool_assign(seg[:,:,z-1], max_prop.slice,
                                         max_prop.image, 0)
                cslice = max_prop.slice
                cimage = max_prop.image
            seg[:,:,min(z,seg.shape[2]-1)] = fill_holes(seg[:,:,min(z,
                                                                    seg.shape[2]-1)],
                                                        self.device)
        else:
            z = start+16
            _, props = label(seg[:,:,z-1], device=self.device, to_numpy=False,
                             connectivity=2, calc_prop=True,
                             top_k_region=1)
            if len(props) > 0:
                max_prop = props[0]
                tensor_slice_bool_assign(seg[:,:,z-1], max_prop.slice,
                                         max_prop.image, 0)
                cslice = max_prop.slice
                cimage = max_prop.image
            seg[:,:,min(z, seg.shape[2]-1)] = fill_holes(seg[:,:,min(z,
                                                                     seg.shape[2]-1)],
                                                         self.device)
        return seg, z-1, cslice, cimage
    
    def gen_vessel(self, imt):
        """
        :param imt: (z, y, x) order
        """
        print('4,gen_vessel')
        if not isinstance(imt, torch.Tensor):
            imt = torch.Tensor(imt).to(torch.int16).to(self.device)
        imt = set_window_wl_ww(imt, 400, 1200)
        
        imt = imt.permute(0, 2, 1) # (z, x, y) order
        
        prob, hp = self.seg_3d(imt)
        
        seg = (prob >= 0.5).to(torch.uint8)        
        seg = seg.permute(1, 2, 0) # (x, y, z) order
        prob = prob.permute(1, 2, 0) # (x, y, z) order
        prob = prob.cpu().numpy()
        
        seg = self.remove_extra_vessel(seg)       
        _, props = label(seg, device=self.device, to_numpy=False,
                          connectivity=2, calc_prop=True)
        for p in props:
            if p.bbox[5] - p.bbox[2] < 16:
                tensor_slice_bool_assign(seg, p.slice, p.image, 0)
        seg = seg.cpu().numpy().astype('uint8')

        torch.cuda.empty_cache()
        if self.heatmap:
            hp[hp < 0] = 0
            hp[hp > 2] = 2
            hp = hp*80
            hp = 180 - hp
            hp = np.where(hp >150,150,hp)
            hp = np.transpose(hp, (1, 2, 0))
            return seg, hp
        else:
            return seg, (1-prob)*100
    
    def seg_vessel(self, imt, net):
        print('5,seg_vessel')
        patch_size = (64, 256, 256)        

        if not isinstance(imt, torch.Tensor):
            imt = torch.Tensor(imt).to(torch.int16).to(self.device)

        #imt = imt.permute(2, 1, 0) # xyz order
        imt = set_window_wl_ww(imt, 400, 1200)
        

        #padding
        vx, vy, vz = imt.shape
        px, py, pz = patch_size
        tx, ty, tz =  max(vx, px), max(vy, py), max(vz, pz)
        rpx = max(0, px - vx)
        rpy = max(0, py - vy)
        rpz = max(0, pz - vz)
        imt = F.pad(imt, (0, rpz, 0, rpy, 0, rpx), 'constant', 0)
                
        coords = get_patch_coords(patch_size, imt.shape)
        
        p_x, p_y, p_z = patch_size
        v_x, v_y, v_z = imt.shape
        
        test_set = Volume3DLoader(imt, coords, patch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        seg = torch.FloatTensor(15, v_x, v_y, v_z).zero_().to(self.device)
        if self.heatmap:
            hp = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
        
        net.to(self.device)
        #self.net_hp.to(self.device)

        for i, (image, coord) in enumerate(test_loader):

            image = image.to(self.device)
            out = net(image)
            pred = F.softmax(out['r'], dim=1)
            
            #out_hp = self.net_hp(image)

            for idx in range(image.size(0)):
                sx, ex = coord[idx][0][0], coord[idx][0][1]
                sy, ey = coord[idx][1][0], coord[idx][1][1]
                sz, ez = coord[idx][2][0], coord[idx][2][1]

                #seg[:, sx:ex, sy:ey, sz:ez] = torch.max(seg[:, sx:ex, sy:ey, sz:ez], pred[idx])
                seg[:, sx:ex, sy:ey, sz:ez] += pred[idx]
                #seg[sx:ex, sy:ey, sz:ez] += pred[idx][1]
                '''
                if self.heatmap:
                    hp[sx:ex, sy:ey, sz:ez] += out['hp'][idx][0]
                    num[sx:ex, sy:ey, sz:ez] += 1
                '''
        net.to(self.cpu_device)
        #self.net_hp.to(self.cpu_device)
        seg_s = seg
        seg = seg.max(0)[1]

        '''
        import nibabel as nib
        img = nib.load('/mnt/users/1003378.nii.gz')
        seg_s = seg_s.max(0)[1].cpu().numpy()
        seg_s = np.transpose(seg_s, (1, 2, 0))
        seg_s = seg_s.astype('float32')
        affine = img.affine
        seg_img = nib.Nifti1Image(seg_s, affine)
        nib.save(seg_img, '/mnt/users/1003378_vessel.nii.gz')
        '''
        seg = seg.permute(2, 1, 0).cpu().numpy()
        '''
        prob = seg.float()
        hp = (hp / num.float()) if self.heatmap else np.zeros_like(prob)
        '''
        torch.cuda.empty_cache()
        
        return seg, seg #hp[:vx, :vy, :vz].permute(2, 1, 0).cpu().numpy()
    
    def gen_vessel_xyz(self, imt):
        """
        :param imt: (z, y, x) order
        """
        print('6,gen_vessel_xyz')
        if not isinstance(imt, torch.Tensor):
            imt = torch.Tensor(imt).to(torch.int16).to(self.device)
        imt = imt.permute(2, 1, 0) # xyz order
        imt = set_window_wl_ww(imt, 400, 1200)
        
        prob_x, hp_x = self.seg_vessel(imt, self.net_x)
        
        prob_y, hp_y = self.seg_vessel(imt.permute(1,2,0),
                                       self.net_y)
        prob_y = prob_y.permute(2,0,1)
        hp_y = hp_y.permute(2,0,1)
        
        prob_z, hp_z = self.seg_vessel(imt.permute(2,0,1),
                                       self.net_z)
        prob_z = prob_z.permute(1,2,0)
        hp_z = hp_z.permute(1,2,0)
        
        prob =  (prob_x + prob_y + prob_z) / 3
       
        hp = torch.max(hp_x, hp_y)
        hp = torch.max(hp, hp_z)
        
        seg = (prob >= 0.5).to(torch.uint8)
        _, props = label(seg, device=self.device, to_numpy=False,
                          connectivity=2, calc_prop=True)
        
        for prop in props:
            dx = prop.bbox[3] - prop.bbox[0]
            dy = prop.bbox[4] - prop.bbox[1]
            dz = prop.bbox[5] - prop.bbox[2]
            
            if max(max(dx, dy), dz) < 32:
                tensor_slice_bool_assign(seg, prop.slice, prop.image, 0)
        
        seg = seg.cpu().numpy().astype('uint8')
        prob = prob.cpu().numpy()

        torch.cuda.empty_cache()
        if self.heatmap:
            hp[hp < 0] = 0
            hp[hp > 2] = 2
            hp = (hp*80)
            hp = 180 - hp
            hp = torch.where(hp >150,150,hp)
            hp = hp.cpu().numpy()
            torch.cuda.empty_cache()
            return seg, hp
        else:
            return seg, (1-prob)*100
    
    def seg_vessel_casecade(self, imt, pmsk, net):
        print('7,seg_vessel_casecade')
        adaptive_patch_size = self.patch_adptive
        patch_size = (64, 256, 256)        
        vx, vy, vz = imt.shape
        #padding
        if adaptive_patch_size:
            vy2 = (vy//16 + 1) * 16 if vy % 16 != 0 else vy
            vz2 = (vz//16 + 1) * 16 if vz % 16 != 0 else vz
            patch_size = (patch_size[0], vy2, vz2)

        px, py, pz = patch_size
        
        tx, ty, tz =  max(vx, px), max(vy, py), max(vz, pz)
        rpx = max(0, px - vx)
        rpy = max(0, py - vy)
        rpz = max(0, pz - vz)
        
        imt = F.pad(imt, (0, rpz, 0, rpy, 0, rpx), 'constant', 0)
        pmsk = F.pad(pmsk, (0, rpz, 0, rpy, 0, rpx), 'constant', 0)        
        
        coords = get_patch_coords(patch_size, imt.shape)
        test_set = Volume3DLoaderCasecade(imt, pmsk, coords, patch_size, use_coord=self.use_coord)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        
        v_x, v_y, v_z = imt.shape
        seg = torch.FloatTensor(self.num_class, v_x, v_y, v_z).zero_().to(self.device)
        if self.heatmap:
            hp = torch.FloatTensor(v_x, v_y, v_z).zero_().to(self.device)
            num = torch.ByteTensor(v_x, v_y, v_z).zero_().to(self.device)
        
        net.to(self.device)
        for i, (image, coord) in enumerate(test_loader):

            image = image.to(torch.float32)
            out = net(image)

            pred = F.softmax(out['r'], dim=1)
            for idx in range(image.size(0)):
                sx, ex = coord[idx][0][0], coord[idx][0][1]
                sy, ey = coord[idx][1][0], coord[idx][1][1]
                sz, ez = coord[idx][2][0], coord[idx][2][1]

                seg[:, sx:ex, sy:ey, sz:ez] = torch.max(seg[:, sx:ex, sy:ey, sz:ez], pred[idx])
                if self.heatmap:
                    hp[sx:ex, sy:ey, sz:ez] += out['hp'][idx][0]
                    num[sx:ex, sy:ey, sz:ez] += 1
                
        net.to(self.cpu_device)
        torch.cuda.empty_cache()
        
        seg, _ = seg[1:, ...].max(0)
        prob = seg.float()
        hp = (hp / num.float()).cpu().numpy() if self.heatmap else np.zeros_like(prob)
        return prob[:vx, :vy, :vz], hp[:vx, :vy, :vz]


    
    def gen_vessel_two_stage(self, imt):
        """
        :param imt: (z, y, x) order
        """
        print('8,gen_vessel_two_stage')
        if not isinstance(imt, torch.Tensor):
            imt = torch.Tensor(imt).to(torch.int16).to(self.device)
        #stage one: global segmentation
        imt = imt.permute(0, 2, 1) # zxy order
        src_shape, dst_shape = imt.shape, (256, 256, 256)
        img_g = tensor_resize(imt,
                              dst_shape, False)
        img_g = set_window_wl_ww(img_g, 400, 1200).float()
        img_g = (img_g / 255.0) * 2.0 - 1.0
        img_g = torch.unsqueeze(img_g, dim=0)
        img_g = torch.unsqueeze(img_g, dim=0)
        self.net.to(self.device)
        out = self.net(img_g)
        prob_g = F.softmax(out['r'], dim=1)
        self.net.to(self.cpu_device) 
        
        #import nibabel as nib
        #bg_nii = nib.Nifti1Image((prob_g[0,0]<0.5).cpu().numpy().astype('uint8'), np.eye(4))
        #nib.save(bg_nii, 'bg1.nii.gz')
        '''
        if self.use_completion:
            # from global to cascade
            _, img_g2c = torch.max(prob_g, 1)
            img_g2c = (img_g2c > 0).unsqueeze(dim=0)
            self.net_g2c.to(self.device)
            out = self.net_g2c({'erase': img_g2c})
            prob_g2c = F.softmax(out['r'], dim=1)
            bg = prob_g2c[0, 0]
            self.net_g2c.to(self.cpu_device)
            bg = tensor_resize(bg, src_shape, False)
            seg_g2c = (bg < 0.5).to(torch.float32)
            bg = self.g2c_post(bg, seg_g2c) if self.use_completion_post else seg_g2c
        else:
        '''
        bg = prob_g[0, 0]
        bg = tensor_resize(bg, src_shape, False)
        bg = (bg < 0.5).to(torch.float32)
        
        
        #bg_nii = nib.Nifti1Image(bg.cpu().numpy().astype('uint8'), np.eye(4))
        #nib.save(bg_nii, 'bg2.nii.gz')
        
        #stage two: cascade segmentation
        imt = set_window_wl_ww(imt, 400, 1200).float()
        prob, hp = self.seg_vessel_casecade(imt, bg, self.net)
        prob = prob.permute(1,2,0)
        hp = np.transpose(hp, (1,2,0))
        seg = (prob >= 0.5).to(torch.uint8)

        seg, sz, cslice, cimage = self.remove_extra_vessel(seg)
        _, props = label(seg, device=self.device, to_numpy=False,
                          connectivity=2, calc_prop=True)
        vz = seg.shape[2]
        for prop in props:
            dx = prop.bbox[3] - prop.bbox[0]
            dy = prop.bbox[4] - prop.bbox[1]
            dz = prop.bbox[5] - prop.bbox[2]
            
            az = (prop.bbox[5] + prop.bbox[2]) // 2
            if az >= (vz * 2 // 3):
                if max(max(dx, dy), dz) < 32:
                    tensor_slice_bool_assign(seg, prop.slice, prop.image, 0)
            elif dz < 32:
                tensor_slice_bool_assign(seg, prop.slice, prop.image, 0)
        
        seg = self.resume_deleted_vessel(seg, sz, cslice, cimage)
        seg = seg.cpu().numpy().astype('uint8')
        prob = prob.cpu().numpy()

        #  import pickle
        #  with open('mask.pickle', 'wb') as fp:
            #  pickle.dump([out1, out2], fp)
        
        torch.cuda.empty_cache()
        if self.heatmap:
            hp[hp < 0] = 0
            hp[hp > 2] = 2
            hp = (hp*80)
            hp = 180 - hp
            hp = np.where(hp >150,150,hp)
            return seg, hp
        else:
            return seg, (1-prob)*100

    def gen_vessel_crop(self, imt):
        print('9,gen_vessel_crop')
        return self.seg_vessel(imt, self.net)
        '''
        if self.mode == 'casecade':
            return self.gen_vessel_two_stage(imt)
        return self.gen_vessel_xyz(imt)
        '''
    def g2c_post(self, vessel1, vessel2):
        print('10,g2c_post')
        stage0_labels, stage0_props = label(vessel1>0, connectivity=1,
                                            device=self.device, to_numpy=True,
                                            calc_prop=True)
        for prop in stage0_props:
            if prop.bbox[3] - prop.bbox[0] < 5:
                tensor_slice_bool_assign(stage0_labels, prop.slice, prop.image,
                                         0)

        vessel2[(vessel1 == 0) & (vessel2 > 0)] = 2
        _, stage1_props = label(vessel2 == 2, device=self.device,
                                to_numpy=False, connectivity=2, calc_prop=True)
        for prop in stage1_props:
            if prop.bbox[3] - prop.bbox[0] < 5:
                tensor_slice_bool_assign(vessel2, prop.slice, prop.image, 0)
            z1, y1, x1, z2, y2, x2 = prop.bbox
            prop_label = stage0_labels[z1:z2, y1:y2, x1:x2]
            if np.unique(prop_label).shape[0] < 3:
                tensor_slice_bool_assign(vessel2, prop.slice, prop.image, 0)
        vessel2[vessel2 == 2] = 1
        return vessel2
