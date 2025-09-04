"""
精确数学配图SVG转换器主模块
实现完整的4层转换流水线，从图像到精确SVG的端到端转换
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import time

from math_config import MathConfig
from geometry_primitives import Point, GeometryPrimitive
from geometry_detector import HybridGeometryDetector
from mathematical_ocr import MathematicalOCR
from constraint_solver import IncrementalConstraintSolver
from math_semantic_analyzer import MathSemanticAnalyzer
from precise_svg_generator import PreciseSVGGenerator


@dataclass
class ConversionResult:
    """转换结果数据结构"""
    success: bool
    svg_content: Optional[str] = None
    svg_path: Optional[Path] = None
    geometry_count: int = 0
    text_count: int = 0
    constraint_count: int = 0
    processing_time: float = 0.0
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    intermediate_results: Optional[Dict[str, Any]] = None


class PreciseMathSVGConverter:
    """精确数学配图SVG转换器主类"""
    
    def __init__(self, config: Optional[MathConfig] = None):
        """初始化转换器"""
        self.config = config or MathConfig()
        self.logger = self._setup_logger()
        
        # 初始化各层组件
        self.geometry_detector = HybridGeometryDetector(self.config.geometry_detection)
        self.ocr_system = MathematicalOCR(self.config.ocr)
        self.constraint_solver = IncrementalConstraintSolver(self.config.constraint_solver)
        self.semantic_analyzer = MathSemanticAnalyzer(self.config.semantic_analysis)
        self.svg_generator = PreciseSVGGenerator(self.config.svg_generation)
        
        self.logger.info("精确数学配图SVG转换器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('PreciseMathSVGConverter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_image_to_svg(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        save_intermediates: bool = False
    ) -> ConversionResult:
        """
        主转换方法：从图像文件转换为精确SVG
        
        Args:
            image_path: 输入图像路径
            output_path: 输出SVG路径（可选）
            save_intermediates: 是否保存中间结果
            
        Returns:
            ConversionResult: 转换结果
        """
        start_time = time.time()
        intermediate_results = {} if save_intermediates else None
        
        try:
            self.logger.info(f"开始转换图像: {image_path}")
            
            # 加载和预处理图像
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return ConversionResult(
                    success=False,
                    error_message=f"无法加载图像: {image_path}",
                    processing_time=time.time() - start_time
                )
            
            if save_intermediates:
                intermediate_results['preprocessed_image'] = image
            
            # Layer 1: 几何检测 + OCR
            self.logger.info("Layer 1: 执行几何检测和OCR...")
            geometry_elements, text_elements = self._layer1_detection(image)
            
            if save_intermediates:
                intermediate_results['geometry_elements'] = geometry_elements
                intermediate_results['text_elements'] = text_elements
            
            # Layer 2: 约束求解
            self.logger.info("Layer 2: 执行约束求解...")
            refined_geometry = self._layer2_constraint_solving(geometry_elements)
            
            if save_intermediates:
                intermediate_results['refined_geometry'] = refined_geometry
            
            # Layer 3: 语义分析
            self.logger.info("Layer 3: 执行语义分析...")
            semantic_scene = self._layer3_semantic_analysis(
                refined_geometry, text_elements, image
            )
            
            if save_intermediates:
                intermediate_results['semantic_scene'] = semantic_scene
            
            # Layer 4: SVG生成
            self.logger.info("Layer 4: 生成精确SVG...")
            svg_content = self._layer4_svg_generation(semantic_scene, image.shape)
            
            # 保存SVG文件
            svg_path = None
            if output_path:
                svg_path = Path(output_path)
                svg_path.write_text(svg_content, encoding='utf-8')
                self.logger.info(f"SVG已保存到: {svg_path}")
            
            # 计算统计信息
            processing_time = time.time() - start_time
            confidence_score = self._calculate_confidence_score(
                geometry_elements, text_elements, refined_geometry
            )
            
            self.logger.info(f"转换完成，用时: {processing_time:.2f}秒")
            
            return ConversionResult(
                success=True,
                svg_content=svg_content,
                svg_path=svg_path,
                geometry_count=len(refined_geometry),
                text_count=len(text_elements),
                constraint_count=self.constraint_solver.get_constraint_count(),
                processing_time=processing_time,
                confidence_score=confidence_score,
                intermediate_results=intermediate_results
            )
            
        except Exception as e:
            self.logger.error(f"转换过程中出错: {str(e)}")
            return ConversionResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                intermediate_results=intermediate_results
            )
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载和预处理图像"""
        try:
            # 使用中文路径兼容的方式读取图像
            import numpy as np
            from pathlib import Path
            
            # 先尝试正常读取
            image = cv2.imread(image_path)
            
            # 如果失败，尝试用numpy读取（支持中文路径）
            if image is None:
                try:
                    # 使用numpy读取文件
                    with open(image_path, 'rb') as f:
                        file_bytes = np.frombuffer(f.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception as e:
                    self.logger.error(f"使用numpy方法读取图像失败: {str(e)}")
                    return None
            
            if image is None:
                return None
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 图像增强和去噪
            if self.config.preprocessing.apply_enhancement:
                image = self._enhance_image(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            return None
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强处理"""
        # 转换为灰度进行处理
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 高斯滤波去噪
        enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
        
        # 转回RGB
        enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        
        return enhanced_image
    
    def _layer1_detection(
        self, 
        image: np.ndarray
    ) -> Tuple[List[GeometryPrimitive], List[Dict]]:
        """Layer 1: 几何检测和OCR"""
        # 几何检测
        geometry_elements = self.geometry_detector.detect_geometries(image)
        self.logger.info(f"检测到 {len(geometry_elements)} 个几何元素")
        
        # OCR文本提取
        text_elements = self.ocr_system.extract_mathematical_text(image, geometry_elements)
        self.logger.info(f"提取到 {len(text_elements)} 个文本元素")
        
        return geometry_elements, text_elements
    
    def _layer2_constraint_solving(
        self, 
        geometry_elements: List[GeometryPrimitive]
    ) -> List[GeometryPrimitive]:
        """Layer 2: 约束求解和几何精化"""
        refined_geometry = self.constraint_solver.solve_constraints(geometry_elements, [])
        constraint_count = self.constraint_solver.get_constraint_count()
        
        self.logger.info(f"应用了 {constraint_count} 个约束，精化了几何元素")
        
        return refined_geometry
    
    def _layer3_semantic_analysis(
        self,
        geometry_elements: List[GeometryPrimitive],
        text_elements: List[Dict],
        image: np.ndarray
    ) -> Dict[str, Any]:
        """Layer 3: 语义分析和场景理解"""
        semantic_scene = self.semantic_analyzer.analyze_mathematical_scene(
            geometry_elements, text_elements, image
        )
        
        self.logger.info("完成语义分析和场景理解")
        
        return semantic_scene
    
    def _layer4_svg_generation(
        self, 
        semantic_scene: Dict[str, Any], 
        image_shape: Tuple[int, ...]
    ) -> str:
        """Layer 4: 精确SVG生成"""
        svg_content = self.svg_generator.generate_precise_svg(
            semantic_scene, 
            image_width=image_shape[1],
            image_height=image_shape[0]
        )
        
        self.logger.info("完成精确SVG生成")
        
        return svg_content
    
    def _calculate_confidence_score(
        self,
        geometry_elements: List[GeometryPrimitive],
        text_elements: List[Dict],
        refined_geometry: List[GeometryPrimitive]
    ) -> float:
        """计算转换置信度分数"""
        # 基础分数
        base_score = 0.5
        
        # 几何检测质量加分
        if geometry_elements:
            geometry_score = min(len(geometry_elements) / 10.0, 0.3)
            base_score += geometry_score
        
        # 文本识别质量加分
        if text_elements:
            text_score = min(len(text_elements) / 5.0, 0.2)
            base_score += text_score
        
        # 约束求解效果加分
        constraint_count = self.constraint_solver.get_constraint_count()
        if constraint_count > 0:
            constraint_score = min(constraint_count / 20.0, 0.2)
            base_score += constraint_score
        
        return min(base_score, 1.0)
    
    def convert_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> List[ConversionResult]:
        """批量转换图像"""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"处理第 {i+1}/{len(image_paths)} 张图像: {image_path}")
            
            # 生成输出文件名
            input_path = Path(image_path)
            output_file = output_path / f"{input_path.stem}.svg"
            
            # 执行转换
            result = self.convert_image_to_svg(
                image_path=image_path,
                output_path=str(output_file)
            )
            results.append(result)
            
            # 调用进度回调
            if progress_callback:
                progress_callback(i + 1, len(image_paths), result)
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息和统计"""
        return {
            'version': '1.0.0',
            'components': {
                'geometry_detector': type(self.geometry_detector).__name__,
                'ocr_system': type(self.ocr_system).__name__,
                'constraint_solver': type(self.constraint_solver).__name__,
                'semantic_analyzer': type(self.semantic_analyzer).__name__,
                'svg_generator': type(self.svg_generator).__name__,
            },
            'config': {
                'precision': self.config.svg_generation.COORDINATE_PRECISION,
                'ocr_engines': len(self.config.ocr.EASYOCR_CONFIG.get('engines', [])),
                'detection_algorithms': 3,  # contour, hough, template
            }
        }


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='精确数学配图SVG转换器')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('-o', '--output', help='输出SVG路径')
    parser.add_argument('--save-intermediates', action='store_true', 
                       help='保存中间处理结果')
    parser.add_argument('--config', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建转换器
    config = MathConfig()
    if args.config:
        # 这里可以添加从文件加载配置的逻辑
        pass
    
    converter = PreciseMathSVGConverter(config)
    
    # 执行转换
    result = converter.convert_image_to_svg(
        image_path=args.input,
        output_path=args.output,
        save_intermediates=args.save_intermediates
    )
    
    # 输出结果
    if result.success:
        print(f"转换成功！")
        print(f"几何元素: {result.geometry_count}")
        print(f"文本元素: {result.text_count}")
        print(f"约束数量: {result.constraint_count}")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"置信度: {result.confidence_score:.2f}")
        if result.svg_path:
            print(f"输出文件: {result.svg_path}")
    else:
        print(f"转换失败: {result.error_message}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())