"""
仅测试几何检测部分
"""

import cv2
import numpy as np
from math_config import MathConfig
from geometry_detector import HybridGeometryDetector
from pathlib import Path

def main():
    print("仅测试几何检测...")
    
    # 创建测试图像
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # 绘制一些简单的几何图形
    cv2.rectangle(test_image, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.circle(test_image, (250, 100), 50, (0, 0, 0), 2)
    cv2.line(test_image, (50, 200), (350, 250), (0, 0, 0), 2)
    
    print("创建了测试图像")
    
    # 创建检测器
    config = MathConfig()
    detector = HybridGeometryDetector(config.geometry_detection)
    
    print("开始几何检测...")
    try:
        geometries = detector.detect_geometries(test_image)
        print(f"检测成功，找到 {len(geometries)} 个几何元素")
        
        for i, geom in enumerate(geometries):
            print(f"  {i+1}. {geom.type} - 置信度: {geom.confidence:.3f}")
            
    except Exception as e:
        import traceback
        print(f"检测失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()