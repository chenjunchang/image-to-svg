"""
验证脚本：转换input_image目录下的PNG图片为SVG
"""

import os
import sys
from pathlib import Path
import time
import logging
from typing import List

from math_config import MathConfig
from precise_math_svg_converter import PreciseMathSVGConverter

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_conversion():
    """验证转换功能"""
    print("=" * 60)
    print("精确数学配图SVG转换系统 - 验证测试")
    print("=" * 60)
    
    setup_logging()
    
    # 检查input_image目录
    input_dir = Path("input_image")
    if not input_dir.exists():
        print("[ERROR] input_image目录不存在")
        return False
    
    # 获取所有PNG文件
    png_files = list(input_dir.glob("*.png"))
    if not png_files:
        print("[ERROR] input_image目录中没有PNG文件")
        return False
    
    print(f"找到 {len(png_files)} 个PNG文件:")
    for i, file in enumerate(png_files, 1):
        print(f"   {i}. {file.name}")
    
    # 创建输出目录
    output_dir = Path("output_svg_validation")
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 创建转换器
    print("\n初始化转换器...")
    config = MathConfig()
    converter = PreciseMathSVGConverter(config)
    
    print("系统信息:")
    info = converter.get_system_info()
    print(f"   版本: {info['version']}")
    print(f"   OCR引擎数: {info['config']['ocr_engines']}")
    print(f"   检测算法数: {info['config']['detection_algorithms']}")
    print(f"   坐标精度: {info['config']['precision']}")
    
    # 转换每个文件
    print(f"\n开始转换 {len(png_files)} 个文件...")
    results = []
    total_start_time = time.time()
    
    for i, png_file in enumerate(png_files, 1):
        print(f"\n--- 处理第 {i}/{len(png_files)} 个文件 ---")
        print(f"输入: {png_file}")
        
        # 生成输出文件名
        output_file = output_dir / f"{png_file.stem}.svg"
        
        # 执行转换
        start_time = time.time()
        result = converter.convert_image_to_svg(
            image_path=str(png_file),
            output_path=str(output_file),
            save_intermediates=True
        )
        
        # 记录结果
        results.append({
            'file': png_file.name,
            'result': result,
            'output': output_file if result.success else None
        })
        
        # 显示结果
        if result.success:
            print(f"[SUCCESS] 转换成功!")
            print(f"   几何元素: {result.geometry_count}")
            print(f"   文本元素: {result.text_count}")
            print(f"   约束数量: {result.constraint_count}")
            print(f"   处理时间: {result.processing_time:.2f}秒")
            print(f"   置信度: {result.confidence_score:.2%}")
            print(f"   输出文件: {output_file}")
            
            # 检查输出文件大小
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"   文件大小: {file_size:,} 字节")
        else:
            print(f"[ERROR] 转换失败: {result.error_message}")
    
    # 统计总结果
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r['result'].success)
    
    print(f"\n{'='*60}")
    print("转换总结")
    print(f"{'='*60}")
    print(f"总文件数: {len(results)}")
    print(f"成功转换: {successful}")
    print(f"失败转换: {len(results) - successful}")
    print(f"成功率: {successful/len(results)*100:.1f}%")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每文件: {total_time/len(results):.2f}秒")
    
    if successful > 0:
        avg_confidence = sum(r['result'].confidence_score for r in results if r['result'].success) / successful
        avg_geometry = sum(r['result'].geometry_count for r in results if r['result'].success) / successful
        avg_text = sum(r['result'].text_count for r in results if r['result'].success) / successful
        
        print(f"平均置信度: {avg_confidence:.2%}")
        print(f"平均几何元素: {avg_geometry:.1f}")
        print(f"平均文本元素: {avg_text:.1f}")
    
    # 详细结果列表
    print(f"\n详细结果:")
    for i, result_info in enumerate(results, 1):
        status = "[OK]" if result_info['result'].success else "[FAIL]"
        confidence = f"{result_info['result'].confidence_score:.1%}" if result_info['result'].success else "N/A"
        print(f"{i}. {status} {result_info['file']} - 置信度: {confidence}")
    
    # 输出文件检查
    print(f"\n输出文件检查:")
    svg_files = list(output_dir.glob("*.svg"))
    for svg_file in svg_files:
        size = svg_file.stat().st_size
        print(f"   {svg_file.name}: {size:,} 字节")
    
    print(f"\n验证完成! 输出文件保存在: {output_dir}")
    
    return successful == len(results)

if __name__ == "__main__":
    success = validate_conversion()
    sys.exit(0 if success else 1)