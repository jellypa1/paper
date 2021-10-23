from __future__ import print_function

import time
#import matplotlib.pyplot as plt
import numpy as np
import init_paths
import demo
import demo_t


OBJECT_CONFIDENCE = 0.9

ckpt_path = './ckpt/exif_final/exif_final.ckpt'
demo_obj = demo_t.Demo_t(ckpt_path=ckpt_path, use_gpu=0, quality=1.0, num_per_dim=30)

demo_ori = demo.Demo(ckpt_path=ckpt_path, use_gpu=0, quality=1.0, num_per_dim=30)

import os
for file in os.listdir("./loc_data"):
    print(os.path.join("/loc_data", file))

'''
while True:
    face_data = np.loadtxt('./loc_data/demo_data', delimiter=' ')
    face_data = face_data[np.where(face_data[:, 4]>OBJECT_CONFIDENCE)[0]]
    db_st = time.time()
    im2, res2 = demo_obj('./images/5609.jpg', dense=False, use_area=True) # No upsampling
    print('DBSCAN run time: %.3f' % (time.time() - db_st))

    all_result = demo_obj.get_var()
    np_result = np.array(all_result)
    save_path = './eva_output/' + image_name + '.out'
    np.savetxt(save_path, np_result)
    '''