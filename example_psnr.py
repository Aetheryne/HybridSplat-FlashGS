import torch
import flash_gaussian_splatting

import os
import sys
import json
import time
from PIL import Image  # 用于图片处理
import numpy as np

def compute_psnr(rendering, gt):
    """
    计算PSNR的辅助函数，在计算前进行必要的预处理。
    """
    # 从计算图中分离张量
    rendering_detached = rendering.detach()
    gt_detached = gt.detach()

    # 强制将数值范围裁剪到[0, 1]
    rendering_clamped = torch.clamp(rendering_detached, 0.0, 1.0)
    gt_clamped = torch.clamp(gt_detached, 0.0, 1.0)

    # 计算均方误差 (MSE)
    mse = torch.mean((rendering_clamped - gt_clamped) ** 2)

    # 如果图像完全相同，MSE为0，PSNR为无穷大
    if mse == 0:
        return float('inf')

    # 计算PSNR
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr.item()

class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(
            scene_path)
        print("num_vertex = %d" % self.num_vertex)
        # 58*4byte
        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6


# class Camera:
#     def __init__(self, camera_json):
#         self.id = camera_json['id']
#         self.img_name = camera_json['img_name']
#         self.width = camera_json['width']
#         self.height = camera_json['height']
#         self.position = torch.tensor(camera_json['position'])
#         self.rotation = torch.tensor(camera_json['rotation'])
#         self.focal_x = camera_json['fx']
#         self.focal_y = camera_json['fy']
#         self.zFar = 100.0
#         self.zNear = 0.01
        
# import numpy as np
class Camera:
    def __init__(self, camera_json):
        downsample_scale = 2.0
        
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        # self.width = camera_json['width']
        # self.height = camera_json['height']
        self.width = int(np.floor(camera_json['width'] / downsample_scale))
        self.height = int(np.floor(camera_json['height'] / downsample_scale))
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx'] / downsample_scale
        self.focal_y = camera_json['fy'] / downsample_scale
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # 40 bytes
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                                camera.width, camera.height, 16, 16,
                                                camera.position, camera.rotation,
                                                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                self.points_xy, self.rgb_depth, self.conic_opacity,
                                                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                self.curr_offset)

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 16, 16,
                                                   self.list_sorting_space,
                                                   self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                   self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.int8)
        flash_gaussian_splatting.ops.render_16x16(num_rendered, camera.width, camera.height,
                                                  self.points_xy, self.rgb_depth, self.conic_opacity,
                                                  self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                                  self.ranges, bg_color, out_color)
        return out_color


def save_png(image, path):
    """
    将渲染结果保存为PNG格式
    """
    # 确保图像在CPU上
    image = image.cpu()
    
    # 转换为numpy数组并调整通道顺序为(H, W, C)
    img_np = image.numpy()
    
    img_np = img_np.astype(np.uint8)
    
    # 创建PIL图像对象并保存为PNG
    img_pil = Image.fromarray(img_np)
    img_pil.save(path, format='PNG')

def load_gt_image(img_path, target_size):
    """加载并预处理真实图像"""
    img = Image.open(img_path).convert('RGB')
    # 调整尺寸以匹配渲染尺寸
    img = img.resize(target_size, Image.LANCZOS)
    img_np = np.array(img) / 255.0  # 归一化到[0,1]
    return torch.tensor(img_np, dtype=torch.float32)

def render_scene(model_path, image_path, test_performance=False):
        
    # scene_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    scene_path = os.path.join(model_path, "point_cloud", "iteration_7000", "point_cloud.ply")
    print(scene_path)
    camera_path = os.path.join(model_path, "cameras.json")
    print(camera_path)
    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black

    scene = Scene(device)
    scene.loadPly(scene_path)

    with open(camera_path, 'r') as camera_file:
        cameras_json = json.loads(camera_file.read())

    image_dir = os.path.join(model_path, "test_out")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
        
    gt_images_dir = image_path  
    if not os.path.exists(gt_images_dir):
        print(f"警告：真实图片目录不存在 {gt_images_dir}")
        gt_images_dir = None

    MAX_NUM_RENDERED = 2 ** 24
    MAX_NUM_TILES = 2 ** 20
    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, MAX_NUM_TILES)
    
    # --- 修改代码：初始化用于存储性能指标的列表 ---
    avg_fpss = []
    mem_usages = [] # 用于记录每次渲染的显存占用
    psnrs = [] # 新增: 用于记录每次渲染的PSNR
    
    for camera_json in cameras_json:
        
        camera = Camera(camera_json)
        print("image name = %s" % camera.img_name)

        # 加载真实图像（如果目录存在）
        gt_image = None
        if gt_images_dir:
            gt_path = os.path.join(gt_images_dir, camera.img_name + '.jpg')
            if os.path.exists(gt_path):
                gt_image = load_gt_image(gt_path, (camera.width, camera.height))
            else:
                print(f"警告：真实图片不存在 {gt_path}")
        
        image = rasterizer.forward(scene, camera, bg_color)  # warm up

        if test_performance:
            n = 10
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                image = rasterizer.forward(scene, camera, bg_color)  # test performance
                # --- 新增代码：在每次渲染后记录显存 ---
                if torch.cuda.is_available():
                    mem_usages.append(torch.cuda.memory_allocated())
            torch.cuda.synchronize()
            t1 = time.time()
            # print("elapsed time = %f ms" % ((t1 - t0) / n * 1000))
            # print("fps = %f" % (n / (t1 - t0)))
            avg_fpss.append(n / (t1 - t0))

        image_path = os.path.join(image_dir, "%s.png" % camera.img_name)
        save_png(image, image_path)
        
        # 计算PSNR（如果真实图片存在）
        if gt_image is not None:
            # 将渲染结果转换为浮点并归一化
            rendered_normalized = image.float() / 255.0
            
            # 确保真实图像在GPU上（如果使用GPU）
            gt_image = gt_image.to(device)
            
            # 计算PSNR
            psnr_value = compute_psnr(rendered_normalized, gt_image)
            psnrs.append(psnr_value)
            # print(f"PSNR for {camera.img_name}: {psnr_value:.2f} dB")
        
    print("Average FPS: ", sum(avg_fpss) / len(avg_fpss))
    
    avg_mem = sum(mem_usages) / len(mem_usages)
    print(f"Average GPU Memory Usage: {avg_mem / 1024**2:.2f} MB")
    
    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"Average PSNR: {avg_psnr:.2f} dB")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        render_scene(model_path, image_path, True)
    else:
        # models_path = "D:\\models"  # https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
        models_path = "D:\\UserData\\Desktop\\science\\3dgs\\gaussian-splatting\\models"
        for entry in os.scandir(models_path):
            if entry.is_dir():
                render_scene(entry.path)
