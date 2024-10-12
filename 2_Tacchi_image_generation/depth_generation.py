#!/usr/bin/env python
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage.filters as fi

# 光源定义
# light_sources = [
#         {'position': [0, 1, 0.25], 'color': (255, 255, 255), 'kd': 0.6, 'ks': 0.5},  # white, top
#         {'position': [-1, 0, 0.25], 'color': (255, 130, 115), 'kd': 0.5, 'ks': 0.3},  # blue, right
#         {'position': [0, -1, 0.25], 'color': (108, 82, 255), 'kd': 0.6, 'ks': 0.4},  # red, bottom
#         {'position': [1, 0, 0.25], 'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.1},  # green, left
#     ]

light_sources = [
        {'position': [0, 1, 0.8], 'color': (255, 255, 255), 'kd': 0.05, 'ks': 0.05},  # white, top
        {'position': [-1, 0, 0.8], 'color': (255, 130, 115), 'kd': 0.005, 'ks': 0.8},  # blue, right
        {'position': [1, 0, 0.8], 'color': (108, 82, 255), 'kd': 0.006, 'ks': 0.4},  # red, left
        {'position': [0, -1, 0.8], 'color': (120, 255, 153), 'kd': 0.001, 'ks': 0.1},  # green, bottom
    ]

background = cv2.imread('/home/yuezk/yzk/Tacchi/2_Tacchi_image_generation/background.png')
px2m_ratio = 5.4347826087e-05 # 像素到米的比例

elastomer_thickness = 0.004 # # 弹性体厚度（米）
min_depth = 0.026 # distance from the image sensor to the rigid glass outer surface # 到传感器的最小深度
max_depth = min_depth + elastomer_thickness # 最大深度（弹性体的底部）

ka = 0.8 # 环境光照系数
default_alpha = 5 # 默认的高光系数

t = 3 # 弹性变形迭代次数
sigma = 7 # 高斯核的标准差
kernel_size = 21 # 高斯核大小

# 添加噪声的函数 该函数向图像添加高斯噪声，用于模拟物体表面的微小随机不规则性。
def gaus_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

# 生成高斯核的函数 生成二维高斯核，用于在图像中进行平滑处理和模拟弹性体变形。
def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

# 计算图像梯度的函数 计算图像在 x 轴或 y 轴上的梯度，用于生成法向量。
def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0

# 法向量计算 根据图像的 x 和 y 轴梯度计算每个像素点的法向量，法向量用于光照计算中的漫反射和镜面反射。
def tangent(mat):
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return (unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))

# 这个函数的主要用途是快速生成一个纯色图像，可以用于图像处理或计算机视觉中的各种应用场景。
def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64) # size + (3,) 表示生成一个形状为 size 的数组，并在最后添加一个维度 3，代表图像的 RGB 三个颜色通道。
    image[:] = color
    return image

# 光照和叠加处理将光照效果叠加到背景图像上。alpha 控制透明度，color 是叠加的颜色，函数将颜色和光照信息合成并返回处理后的图像。
def add_overlay(rgb, alpha, color):
    s = np.shape(alpha) # (480, 640, 3)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))  # * 10.0

    overlay = solid_color_img(color, s) # (480, 640, 3)

    foreground = opacity3 * overlay
    background = (1.0 - opacity3) * rgb.astype(np.float64)
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res

# 分割接触和没有接触的区域
def segments(depth_map):
    case_depth = 20
    not_in_touch = np.copy(depth_map)

    not_in_touch[not_in_touch < max_depth] = 0.0
    not_in_touch[not_in_touch >= max_depth] = 1.0

    in_touch = 1 - not_in_touch

    return not_in_touch, in_touch

def protrusion_map(original, not_in_touch):
    protrusion_map = np.copy(original)
    protrusion_map[not_in_touch >= max_depth] = max_depth
    return protrusion_map

def apply_elastic_deformation_v1(protrusion_depth, not_in_touch, in_touch):
    kernel = gkern2(15, 7)
    deformation = max_depth - protrusion_depth

    for i in range(5):
        deformation = cv2.filter2D(deformation, -1, kernel)
    # return deformation
    return 30 * -deformation * not_in_touch + (protrusion_depth * in_touch)

# 该函数使用高斯平滑核对深度图进行滤波操作，模拟弹性体表面在接触或非接触区域的变形效果。
def apply_elastic_deformation(protrusion_depth, not_in_touch, in_touch):
    protrusion_depth = - (protrusion_depth - max_depth)

    kernel = gkern2(kernel_size, sigma)
    deformation = protrusion_depth

    deformation2 = protrusion_depth
    kernel2 = gkern2(52, 9)

    for i in range(t):
        deformation_ = cv2.filter2D(deformation, -1, kernel)
        r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        deformation = np.maximum(r * deformation_, protrusion_depth)

        deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
        r = np.max(protrusion_depth) / np.max(deformation2_) if np.max(deformation2_) > 0 else 1
        deformation2 = np.maximum(r * deformation2_, protrusion_depth)

    deformation_v1 = apply_elastic_deformation_v1(protrusion_depth, not_in_touch, in_touch)

    for i in range(t):
        deformation_ = cv2.filter2D(deformation2, -1, kernel)
        r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        deformation2 = np.maximum(r * deformation_, protrusion_depth)

    deformation_x = 2 * deformation - deformation2

    return max_depth - deformation_x

# internal_shadow函数有效地将弹性体深度转换为一个标准化的内部阴影深度值。
def internal_shadow(elastomer_depth):
    elastomer_depth_inv = max_depth - elastomer_depth
    elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, elastomer_thickness), (0.0, 1.0))
    return elastomer_depth_inv

# 使用 Phong 着色模型计算光照，包括漫反射 (kd) 和镜面反射 (ks) 分量。T 是表面法向量，source_dir 是光源方向，alpha 是高光的粗糙度参数。
def phong_illumination(T, source_dir, kd, ks, alpha):
    dot = np.dot(T, np.array(source_dir)).astype(np.float64)
    difuse_l = dot * kd
    difuse_l[difuse_l < 0] = 0.0

    dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

    R = 2.0 * dot3 * T - source_dir
    V = [0.0, 0.0, 1.0]

    spec_l = np.power(np.dot(R, V), alpha) * ks
    return difuse_l + spec_l

def display_touch_images(not_in_touch, in_touch):
    # 将 not_in_touch 和 in_touch 转换为 8-bit 图像格式（0-255）
    not_in_touch_display = (not_in_touch * 255).astype(np.uint8)
    in_touch_display = (in_touch * 255).astype(np.uint8)

    # 使用 OpenCV 显示 not_in_touch
    cv2.imshow('Not In Touch', not_in_touch_display)

    # 使用 OpenCV 显示 in_touch
    cv2.imshow('In Touch', in_touch_display)

    # # 等待用户按键来关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 使用 cv2 显示 protrusion_depth
def display_protrusion_depth(protrusion_depth):
    # 归一化 protrusion_depth 到 0-255
    protrusion_depth_normalized = cv2.normalize(protrusion_depth, None, 0, 255, cv2.NORM_MINMAX)

    # 转换为 8-bit 图像格式
    protrusion_depth_display = protrusion_depth_normalized.astype(np.uint8)

    # 使用 OpenCV 显示 protrusion_depth
    cv2.imshow('Protrusion Depth', protrusion_depth_display)

    # # 等待用户按键来关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def display_textured_elastomer_depth(textured_elastomer_depth):
    # 归一化 textured_elastomer_depth 到 0-255
    textured_depth_normalized = cv2.normalize(textured_elastomer_depth, None, 0, 255, cv2.NORM_MINMAX)

    # 转换为 8-bit 图像格式
    textured_depth_display = textured_depth_normalized.astype(np.uint8)

    # 使用 OpenCV 显示 textured_elastomer_depth
    cv2.imshow('Textured Elastomer Depth', textured_depth_display)

def display_tangent_vectors_color(T):
    # 提取法向量的各个分量
    T_x = T[:, :, 0]
    T_y = T[:, :, 1]
    T_z = T[:, :, 2]

    # 归一化到 0-255 的范围
    T_x_normalized = cv2.normalize(T_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    T_y_normalized = cv2.normalize(T_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    T_z_normalized = cv2.normalize(T_z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 合并为 RGB 图像，用颜色表示法向量方向
    tangent_color_image = cv2.merge([T_x_normalized, T_y_normalized, T_z_normalized])

    # 使用 OpenCV 显示法向量方向的彩色图像
    cv2.imshow('Tangent Vectors - Color Coded', tangent_color_image)


def generate(obj_depth):
    not_in_touch, in_touch = segments(obj_depth)
    display_touch_images(not_in_touch, in_touch)

    protrusion_depth = protrusion_map(obj_depth, not_in_touch)
    display_protrusion_depth(protrusion_depth)
    
    elastomer_depth = protrusion_depth

    textured_elastomer_depth = gaus_noise(elastomer_depth, 0.000002)
    display_textured_elastomer_depth(textured_elastomer_depth)

    out = ka * background
    cv2.imshow('ka * Background', out.astype(np.uint8))

    out = add_overlay(out, internal_shadow(protrusion_depth), (0.0, 0.0, 0.0))
    cv2.imshow('Internal Shadow', out.astype(np.uint8))
    T = tangent(textured_elastomer_depth / px2m_ratio)
    display_tangent_vectors_color(T)
    in_touch_expanded = np.repeat(in_touch[:, :, np.newaxis], 3, axis=2)

    for light in light_sources:
        ks = light['ks'] if 'ks' in light else default_ks
        kd = light['kd'] if 'kd' in light else default_kd
        alpha = light['alpha'] if 'alpha' in light else default_alpha
        # add_out = phong_illumination(T, light['position'], kd, ks, alpha)
        # add_out[not_in_touch == 1] = 0
        # out = add_overlay(out, add_out, light['color'])
        out = add_overlay(out, phong_illumination(T, light['position'], kd, ks, alpha), light['color'])
        
    return out