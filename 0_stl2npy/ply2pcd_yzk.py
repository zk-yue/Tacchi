import open3d as o3d

# 加载 PLY 网格文件
mesh = o3d.io.read_triangle_mesh("/home/yuezk/yzk/Tacchi/0_stl2npy/ply/sphere_small.ply")

# 确保法线存在（如果没有，可以重新计算）
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# 使用均匀采样从网格生成 1,000,000 个点的点云
point_cloud = mesh.sample_points_uniformly(number_of_points=1000000)

# 创建 PCD 保存目录（如果没有）
import os
if not os.path.exists('pcd'):
    os.makedirs('pcd')

# 保存点云为 PCD 文件
o3d.io.write_point_cloud("/home/yuezk/yzk/Tacchi/0_stl2npy/pcd/sphere_small.pcd", point_cloud)
print("Point cloud saved to pcd/sphere_small.pcd")
