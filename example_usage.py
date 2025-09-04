"""
精确数学配图SVG转换系统使用示例
演示如何使用完整的转换系统
"""

import cv2
import numpy as np
from pathlib import Path
import logging

from math_config import MathConfig
from precise_math_svg_converter import PreciseMathSVGConverter
from test_framework import TestDataGenerator, TestFramework


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建配置
    config = MathConfig()
    
    # 创建转换器
    converter = PreciseMathSVGConverter(config)
    
    # 生成测试图像
    print("生成测试图像...")
    test_image = TestDataGenerator.generate_synthetic_geometry_image(
        width=800, 
        height=600,
        shapes=['line', 'circle', 'rectangle', 'triangle']
    )
    
    # 保存测试图像
    test_image_path = "test_geometry.png"
    cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    print(f"测试图像已保存到: {test_image_path}")
    
    # 执行转换
    print("执行SVG转换...")
    result = converter.convert_image_to_svg(
        image_path=test_image_path,
        output_path="output_basic.svg",
        save_intermediates=True
    )
    
    # 输出结果
    if result.success:
        print("转换成功！")
        print(f"几何元素数量: {result.geometry_count}")
        print(f"文本元素数量: {result.text_count}")
        print(f"约束数量: {result.constraint_count}")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"置信度: {result.confidence_score:.2%}")
        print(f"SVG输出: {result.svg_path}")
    else:
        print(f"转换失败: {result.error_message}")


def example_math_diagram():
    """数学图表示例"""
    print("\n=== 数学图表转换示例 ===")
    
    config = MathConfig()
    converter = PreciseMathSVGConverter(config)
    
    # 生成包含文本的数学图表
    print("生成数学图表...")
    test_image, annotations = TestDataGenerator.generate_math_diagram_with_text(
        width=800, 
        height=600
    )
    
    # 保存测试图像
    test_image_path = "test_math_diagram.png"
    cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    print(f"数学图表已保存到: {test_image_path}")
    print(f"期望检测: {len(annotations['shapes'])} 个几何元素, {len(annotations['texts'])} 个文本元素")
    
    # 执行转换
    print("执行数学图表转换...")
    result = converter.convert_image_to_svg(
        image_path=test_image_path,
        output_path="output_math_diagram.svg",
        save_intermediates=True
    )
    
    # 输出结果和比较
    if result.success:
        print("转换成功！")
        print(f"检测到几何元素: {result.geometry_count} (期望: {len(annotations['shapes'])})")
        print(f"检测到文本元素: {result.text_count} (期望: {len(annotations['texts'])})")
        print(f"约束数量: {result.constraint_count}")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"置信度: {result.confidence_score:.2%}")
        print(f"SVG输出: {result.svg_path}")
    else:
        print(f"转换失败: {result.error_message}")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    config = MathConfig()
    converter = PreciseMathSVGConverter(config)
    
    # 创建多个测试图像
    test_images = []
    for i in range(3):
        shapes = [['line', 'circle'], ['rectangle', 'triangle'], ['circle', 'line', 'rectangle']][i]
        image = TestDataGenerator.generate_synthetic_geometry_image(
            width=600, 
            height=400,
            shapes=shapes
        )
        
        image_path = f"test_batch_{i+1}.png"
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        test_images.append(image_path)
    
    print(f"创建了 {len(test_images)} 个测试图像")
    
    # 批量处理
    def progress_callback(current, total, result):
        print(f"处理进度: {current}/{total} - {'成功' if result.success else '失败'}")
    
    results = converter.convert_batch(
        image_paths=test_images,
        output_dir="batch_output",
        progress_callback=progress_callback
    )
    
    # 统计结果
    successful = sum(1 for r in results if r.success)
    total_time = sum(r.processing_time for r in results)
    avg_confidence = np.mean([r.confidence_score for r in results if r.success])
    
    print(f"批量处理完成: {successful}/{len(results)} 成功")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均置信度: {avg_confidence:.2%}")


def example_system_testing():
    """系统测试示例"""
    print("\n=== 系统测试示例 ===")
    
    config = MathConfig()
    framework = TestFramework(config)
    
    print("运行完整测试套件...")
    results = framework.run_full_test_suite()
    
    # 保存测试结果
    framework.save_test_results(results, "example_test_results.json")
    
    # 显示测试摘要
    summary = results['summary']
    print("测试摘要:")
    print(f"  总测试数: {summary['total_tests']}")
    print(f"  通过测试: {summary['passed_tests']}")
    print(f"  失败测试: {summary['failed_tests']}")
    print(f"  成功率: {summary['passed_tests']/summary['total_tests']*100:.1f}%")
    print(f"  平均准确率: {summary['average_accuracy']:.2%}")
    print(f"  平均处理时间: {summary['average_processing_time']:.2f}秒")
    
    # 显示组件测试结果
    print("\n组件测试结果:")
    for component, result in results['component_tests'].items():
        status = "✓" if result['success'] else "✗"
        accuracy = result['metrics']['accuracy']
        time_taken = result['metrics']['processing_time']
        print(f"  {status} {component}: 准确率 {accuracy:.2%}, 时间 {time_taken:.3f}s")


def example_system_info():
    """系统信息示例"""
    print("\n=== 系统信息 ===")
    
    config = MathConfig()
    converter = PreciseMathSVGConverter(config)
    
    info = converter.get_system_info()
    
    print("系统版本:", info['version'])
    print("组件:")
    for component, name in info['components'].items():
        print(f"  {component}: {name}")
    
    print("配置:")
    for key, value in info['config'].items():
        print(f"  {key}: {value}")


def cleanup_example_files():
    """清理示例文件"""
    print("\n=== 清理示例文件 ===")
    
    files_to_remove = [
        "test_geometry.png",
        "output_basic.svg",
        "test_math_diagram.png", 
        "output_math_diagram.svg",
        "test_batch_1.png",
        "test_batch_2.png", 
        "test_batch_3.png",
        "example_test_results.json"
    ]
    
    # 清理批量输出目录
    batch_dir = Path("batch_output")
    if batch_dir.exists():
        import shutil
        shutil.rmtree(batch_dir)
        print("删除批量输出目录")
    
    # 清理单个文件
    removed_count = 0
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            removed_count += 1
    
    print(f"清理了 {removed_count} 个示例文件")


def main():
    """主函数"""
    print("精确数学配图SVG转换系统 - 使用示例")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # 显示系统信息
        example_system_info()
        
        # 基本使用示例
        example_basic_usage()
        
        # 数学图表示例
        example_math_diagram()
        
        # 批量处理示例
        example_batch_processing()
        
        # 系统测试示例
        example_system_testing()
        
        print("\n所有示例执行完成！")
        
        # 询问是否清理文件
        user_input = input("\n是否清理生成的示例文件? (y/n): ")
        if user_input.lower() in ['y', 'yes', '是']:
            cleanup_example_files()
        
    except Exception as e:
        print(f"示例执行过程中出错: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())