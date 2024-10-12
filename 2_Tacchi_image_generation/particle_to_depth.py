import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from depth_generation import generate
import cv2
import os
import time
tic = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--object", default="sphere_small")
parser.add_argument("--particle", default="100", choices=["1","10","100"])
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--y", type=int, default=0)
args = parser.parse_args()

if not os.path.exists("unaligned_" + args.particle+ "/sim"):
    os.makedirs("unaligned_" + args.particle+ "/sim")
if not os.path.exists("unaligned_" + args.particle+ "/depth"):
    os.makedirs("unaligned_" + args.particle+ "/depth")

obj_name = '/home/yuezk/yzk/Tacchi/2_Tacchi_image_generation/surface/' + args.particle + '_' +args.object+'.npz'

data = np.load(obj_name) # 加载gel_press.py生成的数据

p_xpos = data['p_xpos_list']
p_ypos = data['p_ypos_list']
p_zpos = data['p_zpos_list']
print(np.shape(p_zpos)) # (128, 10201)

num_particle = 101
# img_length = 640
img_length = 480
img_width = 480

p_depth = np.zeros((num_particle,num_particle)) # 存储粒子的深度信息。
i_depth = np.zeros((img_width,img_length)) # 生成的插值深度图像，大小为 640 x 480。

# xi = np.linspace(12.5, 27.5, num=img_length)
xi = np.linspace(14.375, 25.625, num=img_length)
yi = np.linspace(14.375, 25.625, num=img_width)

z_ref = 0.03
k=0

for i in range(np.shape(p_zpos)[0]):

    # xp=[]
    # yp=[]
    # p_depth=[]


    # for j in range(10201):
    #     if p_xpos[i][j]>=12.5 and p_xpos[i][j]<=27.5 and p_ypos[i][j]>=14.375 and p_ypos[i][j]<=25.625:
    #         xp = np.append(xp,p_xpos[i][j])
    #         yp = np.append(yp,p_ypos[i][j])
    #         p_depth = np.append(p_depth,(p_zpos[i][j]-12)/1000+0.03)

    xp = p_xpos[i][:10201] # 101*101=10201
    yp = p_ypos[i][:10201]
    p_depth = (p_zpos[i][:10201]-12)/1000+0.03

    if np.abs(np.min(p_depth)-z_ref)<1e-4 and z_ref>=0.02:
        # 将离散的粒子深度数据插值到一个规则的网格上。
        # yp 和 xp 分别是数据点的 y 坐标和 x 坐标。
        # p_depth 是这些数据点对应的深度值。
        # xi 和 yi 是插值后的网格的坐标。(xi[None,:],yi[:,None])将一维数组 xi 和 yi 转换为二维网格坐标。
        # method='cubic' 指定了插值方法为三次插值，这种方法在二维情况下会返回一个分段三次、多项式连续可微的插值结果。fill_value=0.03 则指定了在插值范围之外的点的默认填充值为 0.03。
        i_depth = interpolate.griddata((yp, xp), p_depth, (xi[None,:],yi[:,None]), method='cubic', fill_value=0.03)

        i_depth = i_depth.astype(np.float32)
        print("最大",i_depth.max(),"最小",i_depth.min())
        pos = "__"+str((args.y+1)*33+(args.x+1)*11+k+1)+"__%d_%d_%d"%(args.y,args.x,k)

        npy_name = "unaligned_" + args.particle+ "/depth/" +args.object+pos+".npy"
        np.save(npy_name,i_depth)
        i_depth_normalized = cv2.normalize(i_depth, None, 0, 255, cv2.NORM_MINMAX)
        i_depth_normalized = np.uint8(i_depth_normalized)
        cv2.imshow("Depth Map", i_depth_normalized)
        # cv2.waitKey(0) 
        img = generate(i_depth)

        img_name = "/home/yuezk/yzk/Tacchi/2_Tacchi_image_generation/unaligned_" + args.particle+ "/sim/" +args.object+pos+".png"

        cv2.imwrite(img_name, img)
        cv2.imshow("unaligned", img)
        cv2.waitKey(0) 

        print(k)
        print(z_ref)
        
        z_ref -=0.0001
        k +=1

toc = time.time()
shijian = toc-tic
print(shijian)