#!/usr/bin/env python3
"""
数学题配图批量转换SVG工具

使用OpenCV轮廓检测技术，将数学题目中的几何图形转换为高质量的SVG矢量图。
专门针对数学配图进行优化，能够准确识别线条、文字和几何形状。
"""

import os
import cv2
import numpy as np
import svgwrite
from pathlib import Path
from PIL import Image, ImageFilter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathImageToSVG:
    def __init__(self, input_dir="input_image", output_dir="output_svg"):
        """
        初始化转换器
        
        Args:
            input_dir: 输入图片目录
            output_dir: 输出SVG目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # SVG优化参数
        self.svg_precision = 2  # 坐标精度
        self.min_contour_area = 50  # 最小轮廓面积，过滤噪点
        self.approx_epsilon_factor = 0.02  # 轮廓近似精度因子
        
    def preprocess_image(self, image_path):
        """
        图像预处理：针对数学配图优化
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            tuple: (原图, 预处理后的二值图)
        """
        # 使用PIL加载图像
        pil_img = Image.open(image_path)
        
        # 转换为灰度图
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        # 转换为OpenCV格式
        img_array = np.array(pil_img)
        
        # 使用Otsu阈值处理 - 对数学图形的黑白分离效果更好
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 反转图像：让线条变成白色，背景变成黑色（便于轮廓检测）
        binary = cv2.bitwise_not(binary)
        
        # 形态学操作：连接断开的线条，去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 去除太小的噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        return img_array, binary
    
    def find_contours(self, binary_image):
        """
        查找轮廓并进行筛选
        
        Args:
            binary_image: 二值化图像
            
        Returns:
            list: 筛选后的轮廓列表
        """
        # 查找所有轮廓，包括内部轮廓（适合数学图形中的文字和符号）
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_TREE,  # 检测所有轮廓并建立层次关系
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 筛选轮廓：去除太小的噪点和太大的背景
        filtered_contours = []
        image_area = binary_image.shape[0] * binary_image.shape[1]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 过滤条件：
            # 1. 面积大于最小阈值
            # 2. 面积小于图像面积的70%（排除背景）
            # 3. 轮廓周长大于10像素（排除单点噪声）
            perimeter = cv2.arcLength(contour, True)
            
            if (area > self.min_contour_area and 
                area < image_area * 0.7 and 
                perimeter > 10):
                
                # 轮廓近似：减少不必要的点，保持形状
                epsilon = self.approx_epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 确保近似后还有足够的点
                if len(approx) >= 3:
                    filtered_contours.append(approx)
                
        return filtered_contours
    
    def create_svg_from_contours(self, contours, image_shape, output_path):
        """
        根据轮廓创建SVG文件
        
        Args:
            contours: 轮廓列表
            image_shape: 原图尺寸 (height, width)
            output_path: 输出SVG路径
        """
        height, width = image_shape
        
        # 创建SVG画布
        dwg = svgwrite.Drawing(
            str(output_path),
            size=(f'{width}px', f'{height}px'),
            viewBox=f'0 0 {width} {height}'
        )
        
        # 添加样式
        dwg.add(dwg.style("""
            .math-line {
                fill: none;
                stroke: black;
                stroke-width: 2;
                stroke-linejoin: round;
                stroke-linecap: round;
            }
            .math-fill {
                fill: black;
                stroke: black;
                stroke-width: 1;
            }
        """))
        
        # 处理每个轮廓
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue
                
            # 判断是线条还是填充形状
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 计算形状复杂度
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
            else:
                compactness = 0
                
            # 构建路径
            path_data = []
            
            # 起始点
            start_point = contour[0][0]
            path_data.append(f"M {start_point[0]:.{self.svg_precision}f},{start_point[1]:.{self.svg_precision}f}")
            
            # 连接后续点
            for point in contour[1:]:
                x, y = point[0]
                path_data.append(f"L {x:.{self.svg_precision}f},{y:.{self.svg_precision}f}")
            
            # 闭合路径
            path_data.append("Z")
            
            # 创建路径元素
            path_string = " ".join(path_data)
            
            # 根据形状特征决定样式
            if compactness > 0.3 and area > 500:
                # 较规整的形状，使用填充
                css_class = "math-fill"
            else:
                # 线条或不规整形状，使用描边
                css_class = "math-line"
                
            path = dwg.path(
                d=path_string,
                class_=css_class,
                id=f"contour_{i}"
            )
            
            dwg.add(path)
        
        # 保存SVG文件
        dwg.save()
        logger.info(f"SVG已保存: {output_path}")
        
    def convert_image(self, image_path):
        """
        转换单个图像文件
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            logger.info(f"开始处理: {image_path}")
            
            # 预处理
            original, binary = self.preprocess_image(image_path)
            
            # 查找轮廓
            contours = self.find_contours(binary)
            logger.info(f"找到 {len(contours)} 个有效轮廓")
            
            # 生成输出文件名
            stem = Path(image_path).stem
            output_path = self.output_dir / f"{stem}.svg"
            
            # 创建SVG
            self.create_svg_from_contours(contours, original.shape, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"处理 {image_path} 时出错: {str(e)}")
            return False
    
    def batch_convert(self):
        """
        批量转换input_image目录中的所有图片
        
        Returns:
            dict: 转换结果统计
        """
        if not self.input_dir.exists():
            logger.error(f"输入目录不存在: {self.input_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        # 支持的图片格式
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # 查找所有图片文件
        image_files = []
        for fmt in supported_formats:
            image_files.extend(self.input_dir.glob(f"*{fmt}"))
            image_files.extend(self.input_dir.glob(f"*{fmt.upper()}"))
        
        if not image_files:
            logger.warning(f"在 {self.input_dir} 中未找到支持的图片文件")
            return {"success": 0, "failed": 0, "total": 0}
        
        logger.info(f"找到 {len(image_files)} 个图片文件，开始批量转换...")
        
        # 批量处理
        success_count = 0
        failed_count = 0
        
        for image_file in image_files:
            if self.convert_image(image_file):
                success_count += 1
            else:
                failed_count += 1
        
        # 统计结果
        total = len(image_files)
        logger.info(f"批量转换完成！成功: {success_count}, 失败: {failed_count}, 总计: {total}")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": total
        }

def main():
    """主函数"""
    print("=== 数学题配图批量转换SVG工具 ===")
    print("专门针对数学几何图形进行优化的矢量化转换")
    print()
    
    # 创建转换器
    converter = MathImageToSVG()
    
    # 执行批量转换
    result = converter.batch_convert()
    
    # 显示结果
    print("\n=== 转换结果 ===")
    print(f"成功转换: {result['success']} 个文件")
    print(f"转换失败: {result['failed']} 个文件")
    print(f"总计文件: {result['total']} 个")
    
    if result['success'] > 0:
        print(f"\nSVG文件已保存到: {converter.output_dir}")
        print("建议使用浏览器或SVG编辑器查看转换效果")

if __name__ == "__main__":
    main()