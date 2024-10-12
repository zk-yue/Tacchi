import open3d as o3d
import numpy as np

# 加载3D网格模型 (例如 .obj, .stl 文件)
mesh = o3d.io.read_triangle_mesh("/home/yuezk/yzk/Tacchi/0_stl2npy/stl/sphere_small.stl")

# 确保法线存在（如果没有，可以重新计算法线）
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# 使用均匀采样从网格生成点云
# 你可以根据需要调整采样点的数量
point_cloud = mesh.sample_points_uniformly(number_of_points=1000000)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])

# 保存点云为 PCD 文件
o3d.io.write_point_cloud("/home/yuezk/yzk/Tacchi/0_stl2npy/pcd/sphere_small.pcd", point_cloud)
print("Point cloud saved to output.pcd")
