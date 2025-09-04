#!/usr/bin/env python3
"""
数学图表配置文件
定义系统中使用的各种参数、常量和配置选项
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class GeometryDetectionConfig:
    """几何检测相关配置"""
    
    # 轮廓检测参数
    CONTOUR_PARAMS = {
        'epsilon_factor': 0.02,     # Douglas-Peucker精度因子
        'min_area': 100,            # 最小图形面积
        'min_perimeter': 50,        # 最小图形周长
        'max_vertices': 12,         # 最大顶点数（防止过度复杂化）
    }
    
    # Hough变换参数
    HOUGH_PARAMS = {
        'lines': {
            'threshold': 100,        # 累加器阈值
            'min_line_length': 50,   # 最小线段长度
            'max_line_gap': 10,      # 最大线段间隙
            'rho': 1,               # 距离精度（像素）
            'theta': np.pi/180,     # 角度精度（弧度）
        },
        'circles': {
            'dp': 1,                # 累加器分辨率倍数
            'min_dist': 30,         # 圆心之间最小距离
            'param1': 50,           # Canny边缘检测高阈值
            'param2': 30,           # 累加器阈值
            'min_radius': 10,       # 最小圆半径
            'max_radius': 200,      # 最大圆半径
        }
    }
    
    # 边缘检测参数
    CANNY_PARAMS = {
        'low_threshold': 50,        # 低阈值
        'high_threshold': 150,      # 高阈值
        'aperture_size': 3,         # Sobel核大小
        'l2_gradient': False,       # 使用L1还是L2梯度
    }
    
    # 形状分类阈值
    SHAPE_CLASSIFICATION = {
        'triangle_vertices': 3,
        'rectangle_vertices': 4,
        'angle_tolerance': 15,      # 角度容忍度（度）
        'aspect_ratio_tolerance': 0.2,  # 长宽比容忍度
    }

class OCRConfig:
    """OCR识别相关配置"""
    
    # EasyOCR配置
    EASYOCR_CONFIG = {
        'languages': ['en'],        # 支持语言
        'gpu': True,               # 使用GPU加速
        'confidence_threshold': 0.5, # 置信度阈值
    }
    
    # Tesseract配置
    TESSERACT_CONFIG = {
        'config': '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()+-=∠△∥⊥°',
        'timeout': 10,              # 超时时间（秒）
    }
    
    # 数学符号映射
    MATH_SYMBOLS = {
        'triangle': ['△', 'triangle', '三角形'],
        'angle': ['∠', 'angle', '角'],
        'parallel': ['∥', 'parallel', '平行'],
        'perpendicular': ['⊥', 'perpendicular', '垂直'],
        'degree': ['°', 'degree', '度'],
        'equal': ['=', 'equal', '等于'],
        'plus': ['+', 'plus', '加'],
        'minus': ['-', 'minus', '减'],
    }
    
    # 文本区域提取参数
    TEXT_EXTRACTION = {
        'min_text_height': 10,      # 最小文本高度
        'max_text_height': 100,     # 最大文本高度
        'text_padding': 5,          # 文本区域边距
    }

class ConstraintSolverConfig:
    """约束求解相关配置"""
    
    # 约束类型权重
    CONSTRAINT_WEIGHTS = {
        'parallel': 1.0,
        'perpendicular': 1.0,
        'intersection': 1.2,
        'distance': 0.8,
        'angle': 1.0,
        'collinear': 0.9,
    }
    
    # 优化参数
    OPTIMIZATION_PARAMS = {
        'method': 'L-BFGS-B',      # 优化方法
        'max_iterations': 1000,     # 最大迭代次数
        'tolerance': 1e-6,          # 收敛容差
        'step_size': 0.01,          # 步长
    }
    
    # 约束容差
    CONSTRAINT_TOLERANCE = {
        'parallel_angle': 5.0,      # 平行约束角度容差（度）
        'perpendicular_angle': 5.0, # 垂直约束角度容差（度）
        'distance_tolerance': 2.0,  # 距离容差（像素）
        'position_tolerance': 1.0,  # 位置容差（像素）
    }
    
    # 优先级定义
    CONSTRAINT_PRIORITY = {
        'critical': ['intersection', 'collinear'],
        'important': ['parallel', 'perpendicular', 'angle'],
        'optional': ['distance', 'alignment'],
    }

class SVGGenerationConfig:
    """SVG生成相关配置"""
    
    # SVG文档配置
    SVG_DOCUMENT = {
        'width': 800,
        'height': 600,
        'viewBox': '0 0 800 600',
        'xmlns': 'http://www.w3.org/2000/svg',
        'profile': 'full',
    }
    
    # 坐标精度
    COORDINATE_PRECISION = 6
    
    # 层次结构
    SVG_LAYERS = ['background', 'geometry', 'text', 'annotations', 'debug']
    
    # 样式配置
    STYLES = {
        'geometry': {
            'stroke': 'black',
            'stroke-width': 1,
            'fill': 'none',
        },
        'text': {
            'font-family': 'Arial, sans-serif',
            'font-size': 12,
            'fill': 'black',
        },
        'annotations': {
            'stroke': 'blue',
            'stroke-width': 0.5,
            'fill': 'blue',
            'opacity': 0.7,
        },
        'debug': {
            'stroke': 'red',
            'stroke-width': 0.5,
            'stroke-dasharray': '2,2',
            'opacity': 0.3,
        }
    }

class QualityControlConfig:
    """质量控制相关配置"""
    
    # 精度要求
    ACCURACY_REQUIREMENTS = {
        'geometric_detection': 0.85,    # 几何检测精度要求
        'ocr_recognition': 0.90,        # OCR识别精度要求
        'constraint_solving': 0.95,     # 约束求解精度要求
        'end_to_end': 0.95,            # 端到端转换成功率
    }
    
    # 性能要求
    PERFORMANCE_REQUIREMENTS = {
        'max_processing_time': 5.0,     # 最大处理时间（秒）
        'max_memory_usage': 500,        # 最大内存使用（MB）
    }
    
    # 验证阈值
    VALIDATION_THRESHOLDS = {
        'geometric_error': 0.01,        # 几何误差阈值
        'coordinate_precision': 1e-6,   # 坐标精度阈值
        'angle_precision': 1.0,         # 角度精度阈值（度）
    }

class SystemConfig:
    """系统级配置"""
    
    # 日志配置
    LOGGING = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'precise_math_svg.log',
    }
    
    # 临时文件配置
    TEMP_CONFIG = {
        'temp_dir': 'temp',
        'cleanup_on_exit': True,
        'max_temp_files': 100,
    }
    
    # 并行处理配置
    PARALLEL_CONFIG = {
        'max_workers': 4,               # 最大工作线程数
        'enable_multiprocessing': True, # 启用多进程
    }

# 全局配置实例
GEOMETRY_CONFIG = GeometryDetectionConfig()
OCR_CONFIG = OCRConfig()
CONSTRAINT_CONFIG = ConstraintSolverConfig()
SVG_CONFIG = SVGGenerationConfig()
QUALITY_CONFIG = QualityControlConfig()
SYSTEM_CONFIG = SystemConfig()

def get_config(config_type: str) -> Any:
    """
    获取指定类型的配置
    
    Args:
        config_type: 配置类型 ('geometry', 'ocr', 'constraint', 'svg', 'quality', 'system')
        
    Returns:
        对应的配置对象
    """
    config_map = {
        'geometry': GEOMETRY_CONFIG,
        'ocr': OCR_CONFIG,
        'constraint': CONSTRAINT_CONFIG,
        'svg': SVG_CONFIG,
        'quality': QUALITY_CONFIG,
        'system': SYSTEM_CONFIG,
    }
    
    return config_map.get(config_type.lower())

def update_config(config_type: str, updates: Dict[str, Any]) -> bool:
    """
    更新指定配置
    
    Args:
        config_type: 配置类型
        updates: 要更新的配置字典
        
    Returns:
        是否更新成功
    """
    try:
        config = get_config(config_type)
        if config is None:
            return False
            
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return True
    except Exception as e:
        print(f"配置更新失败: {e}")
        return False

# 配置验证函数
def validate_config() -> List[str]:
    """
    验证配置的合理性
    
    Returns:
        验证错误列表
    """
    errors = []
    
    # 验证几何检测配置
    if GEOMETRY_CONFIG.CONTOUR_PARAMS['epsilon_factor'] <= 0:
        errors.append("epsilon_factor必须大于0")
        
    if GEOMETRY_CONFIG.CONTOUR_PARAMS['min_area'] <= 0:
        errors.append("min_area必须大于0")
    
    # 验证SVG配置
    if SVG_CONFIG.COORDINATE_PRECISION < 1:
        errors.append("坐标精度必须至少为1")
        
    if SVG_CONFIG.SVG_DOCUMENT['width'] <= 0 or SVG_CONFIG.SVG_DOCUMENT['height'] <= 0:
        errors.append("SVG文档尺寸必须大于0")
    
    # 验证质量控制配置
    for key, value in QUALITY_CONFIG.ACCURACY_REQUIREMENTS.items():
        if not 0 <= value <= 1:
            errors.append(f"精度要求 {key} 必须在0-1之间")
    
    return errors

if __name__ == "__main__":
    # 配置验证测试
    validation_errors = validate_config()
    if validation_errors:
        print("配置验证失败:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("所有配置验证通过")
        
    # 配置信息展示
    print(f"\n几何检测配置:")
    print(f"  轮廓检测精度因子: {GEOMETRY_CONFIG.CONTOUR_PARAMS['epsilon_factor']}")
    print(f"  最小图形面积: {GEOMETRY_CONFIG.CONTOUR_PARAMS['min_area']}")
    
    print(f"\nOCR配置:")
    print(f"  支持语言: {OCR_CONFIG.EASYOCR_CONFIG['languages']}")
    print(f"  置信度阈值: {OCR_CONFIG.EASYOCR_CONFIG['confidence_threshold']}")
    
    print(f"\nSVG生成配置:")
    print(f"  文档尺寸: {SVG_CONFIG.SVG_DOCUMENT['width']}x{SVG_CONFIG.SVG_DOCUMENT['height']}")
    print(f"  坐标精度: {SVG_CONFIG.COORDINATE_PRECISION}")


class MathConfig:
    """统一的数学配图转换系统配置类"""
    
    def __init__(self):
        """初始化所有配置"""
        self.geometry_detection = GeometryDetectionConfig()
        self.ocr = OCRConfig()
        self.constraint_solver = ConstraintSolverConfig()
        self.svg_generation = SVGGenerationConfig()
        self.quality_control = QualityControlConfig()
        self.system = SystemConfig()
        self.preprocessing = PreprocessingConfig()
        self.semantic_analysis = SemanticAnalysisConfig()
    
    def validate(self) -> List[str]:
        """验证配置"""
        return validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'geometry_detection': vars(self.geometry_detection),
            'ocr': vars(self.ocr),
            'constraint_solver': vars(self.constraint_solver),
            'svg_generation': vars(self.svg_generation),
            'quality_control': vars(self.quality_control),
            'system': vars(self.system),
        }


class PreprocessingConfig:
    """图像预处理配置"""
    
    def __init__(self):
        self.apply_enhancement = True
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (8, 8)
        self.gaussian_blur_kernel = (3, 3)
        self.gaussian_blur_sigma = 0


class SemanticAnalysisConfig:
    """语义分析配置"""
    
    def __init__(self):
        # 数学模式识别
        self.PATTERN_DETECTION = {
            'enable_geometric_patterns': True,
            'enable_algebraic_patterns': True,
            'enable_coordinate_patterns': True,
            'pattern_confidence_threshold': 0.7,
        }
        
        # 场景图构建
        self.SCENE_GRAPH = {
            'max_nodes': 100,
            'max_edges': 200,
            'relationship_threshold': 0.6,
        }
        
        # 几何-文本绑定
        self.TEXT_GEOMETRY_BINDING = {
            'max_binding_distance': 50,    # 像素
            'confidence_threshold': 0.5,
            'enable_semantic_matching': True,
        }