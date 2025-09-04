"""
快速测试脚本：只转换一个文件验证系统工作
"""

from math_config import MathConfig
from precise_math_svg_converter import PreciseMathSVGConverter
from pathlib import Path

def main():
    print("快速测试：转换单个PNG文件为SVG")
    print("=" * 40)
    
    # 创建转换器
    config = MathConfig()
    converter = PreciseMathSVGConverter(config)
    
    # 查找第一个PNG文件
    input_dir = Path("input_image")
    png_files = list(input_dir.glob("*.png"))
    
    if not png_files:
        print("没有找到PNG文件")
        return
    
    test_file = png_files[0]
    print(f"测试文件: {test_file}")
    
    # 执行转换
    result = converter.convert_image_to_svg(
        image_path=str(test_file),
        output_path="test_output.svg",
        save_intermediates=True
    )
    
    if result.success:
        print("[SUCCESS] 转换成功!")
        print(f"几何元素: {result.geometry_count}")
        print(f"文本元素: {result.text_count}")
        print(f"约束数量: {result.constraint_count}")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"置信度: {result.confidence_score:.2%}")
        print("SVG文件已保存为: test_output.svg")
        
        # 检查文件大小
        output_path = Path("test_output.svg")
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"输出文件大小: {size} 字节")
    else:
        print("[ERROR] 转换失败:")
        print(f"错误: {result.error_message}")

if __name__ == "__main__":
    main()