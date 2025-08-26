import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from glob import glob
import torch

def create_mask(snowflake_image):
    # 打开雪花图像
    # snowflake_image = Image.open(image_path)

    # 将雪花图像转换为灰度图
    snowflake_gray = snowflake_image.convert("L")

    # 将灰度图中像素为黑色的区域作为mask
    threshold = 100  # 调整阈值以适应雪花图像的黑色区域
    mask = snowflake_gray.point(lambda p: p > threshold and 255)

    return mask

def overlay_snow(image_path, snowflake_path, output_path):
    # 打开原始图像
    original_image = Image.open(image_path)
    H, W = original_image.size
    snow = Image.open(snowflake_path).resize((H, W))

    # 将原始图像和mask都转换为numpy数组，以便处理
    original_np = np.array(original_image)
    
    # # print(snow.shape)
    # mask = np.sum(snow) > 0
    mask = create_mask(snow)
    snow = np.array(snow)
    mask = np.array(mask)

    # print(mask.shape, original_np.shape)
    # 将mask中非黑色部分的像素复制到原始图像的对应位置
    for i in range(original_np.shape[0]):
        for j in range(original_np.shape[1]):
            if mask[i, j] != 0:
                original_np[i, j] = snow[i, j]

    # 从numpy数组创建新的图像
    new_image = Image.fromarray(original_np)

    # 保存新图像
    new_image.save(output_path)

def main(input_folder, output_folder, snowflake_folder):
    # 获取输入文件夹中的所有图像文件
    input_files = glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png"))
    snow_files = glob(os.path.join(snowflake_folder, "*.jpg")) + glob(os.path.join(snowflake_folder, "*.png"))
    # 遍历输入文件夹中的每个图像文件
    for input_file in tqdm(input_files, desc="Processing images", ncols=80):
        # 构建输出文件路径
        output_file = os.path.join(output_folder, os.path.basename(input_file))

        index = torch.randint(0, len(snow_files), (1,)).item()
        snowflake_path = snow_files[index]
        # 对每张图像进行处理
        overlay_snow(input_file, snowflake_path, output_file)

    print("Image processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay snowflake images on a batch of images.")
    parser.add_argument("--input_folder", type=str, help="Input folder containing images.")
    parser.add_argument("--output_folder", type=str, help="Output folder for processed images.")
    parser.add_argument("--snowflake_path", type=str, help="Path to the snowflake image.")

    args = parser.parse_args()

    # 创建输出文件夹，如果不存在的话
    os.makedirs(args.output_folder, exist_ok=True)

    # 处理输入文件夹中的图像，并将结果保存到输出文件夹中
    main(args.input_folder, args.output_folder, args.snowflake_path)


# python fuse_img_dir.py --input_folder ../../data/DDN/train/rainy_image --output_folder ../../data/DDN_snow/train/rain_snow_normlight --snowflake_path ../../data/Snow100K-S-mask
# python fuse_img_dir.py --input_folder ../../data/DDN/test/rainy_image --output_folder ../../data/DDN_snow/test/rain_snow_normlight --snowflake_path ../../data/Snow100K-S-mask

# python fuse_img_dir.py --input_folder ../../data/DDN/train/rainy_image --output_folder ../../data/DDN_snow/train/rain_snowL_normlight --snowflake_path ../../data/Snow100K-mask
# python fuse_img_dir.py --input_folder ../../data/DDN/test/rainy_image --output_folder ../../data/DDN_snow/test/rain_snowL_normlight --snowflake_path ../../data/Snow100K-mask

# python fuse_img_dir.py --input_folder ../../data/Raindrop/train/data --output_folder ../../data/Raindrop_snow/train/raindrop_snow_normlight --snowflake_path ../../data/Snow100K-S-mask
# python fuse_img_dir.py --input_folder ../../data/Raindrop/test/data --output_folder ../../data/Raindrop_snow/test/raindrop_snow_normlight --snowflake_path ../../data/Snow100K-S-mask

# python fuse_img_dir.py --input_folder ../../data/Raindrop/train/data --output_folder ../../data/Raindrop_snow/train/raindrop_snowL_normlight --snowflake_path ../../data/Snow100K-mask
# python fuse_img_dir.py --input_folder ../../data/Raindrop/test/data --output_folder ../../data/Raindrop_snow/test/raindrop_snowL_normlight --snowflake_path ../../data/Snow100K-mask