#!/usr/bin/env python3
"""
数学图表OCR系统
实现多引擎OCR、几何掩码引导的文本提取和数学符号识别
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import logging
import re
import subprocess
import tempfile
import os
from dataclasses import dataclass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR未安装，将仅使用Tesseract")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract未安装，将仅使用EasyOCR")

from geometry_primitives import Point, GeometryPrimitive, TextElement
from math_config import OCR_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    bounding_box: Tuple[Point, Point]
    engine: str
    
    def get_center(self) -> Point:
        """获取文本中心点"""
        min_point, max_point = self.bounding_box
        return Point((min_point.x + max_point.x) / 2, 
                    (min_point.y + max_point.y) / 2)

class BaseOCREngine:
    """OCR引擎基类"""
    
    def __init__(self, engine_name: str):
        self.name = engine_name
        
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """提取文本"""
        raise NotImplementedError
        
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """OCR预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised

class EasyOCREngine(BaseOCREngine):
    """EasyOCR引擎"""
    
    def __init__(self):
        super().__init__("easyocr")
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR不可用")
            
        try:
            self.reader = easyocr.Reader(
                OCR_CONFIG.EASYOCR_CONFIG['languages'],
                gpu=OCR_CONFIG.EASYOCR_CONFIG['gpu']
            )
            logger.info("EasyOCR引擎初始化成功")
        except Exception as e:
            logger.error(f"EasyOCR初始化失败: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """使用EasyOCR提取文本"""
        try:
            processed_img = self.preprocess_for_ocr(image)
            results = self.reader.readtext(processed_img)
            
            ocr_results = []
            for result in results:
                # EasyOCR返回格式: (bbox, text, confidence)
                bbox, text, confidence = result
                
                if confidence < OCR_CONFIG.EASYOCR_CONFIG['confidence_threshold']:
                    continue
                    
                # 转换边界框格式
                points = np.array(bbox)
                min_x, min_y = np.min(points, axis=0)
                max_x, max_y = np.max(points, axis=0)
                
                bounding_box = (Point(min_x, min_y), Point(max_x, max_y))
                
                ocr_result = OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    bounding_box=bounding_box,
                    engine=self.name
                )
                ocr_results.append(ocr_result)
                
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR文本提取失败: {e}")
            return []

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR引擎"""
    
    def __init__(self):
        super().__init__("tesseract")
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract不可用")
            
        # 测试Tesseract是否可用
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract引擎初始化成功")
        except Exception as e:
            logger.error(f"Tesseract初始化失败: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """使用Tesseract提取文本"""
        try:
            processed_img = self.preprocess_for_ocr(image)
            
            # 获取详细信息
            data = pytesseract.image_to_data(
                processed_img,
                config=OCR_CONFIG.TESSERACT_CONFIG['config'],
                output_type=pytesseract.Output.DICT,
                timeout=OCR_CONFIG.TESSERACT_CONFIG['timeout']
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0  # 转换为0-1范围
                
                if not text or confidence < 0.3:  # 过滤低质量结果
                    continue
                    
                # 获取边界框
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bounding_box = (Point(x, y), Point(x + w, y + h))
                
                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    engine=self.name
                )
                ocr_results.append(ocr_result)
                
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract文本提取失败: {e}")
            return []

class MathSymbolProcessor:
    """数学符号处理器"""
    
    def __init__(self):
        self.symbol_mappings = OCR_CONFIG.MATH_SYMBOLS
        self.correction_rules = self._build_correction_rules()
        
    def _build_correction_rules(self) -> Dict[str, str]:
        """构建字符纠错规则"""
        return {
            # 常见OCR错误纠正
            '0': 'O',  # 数字0可能被识别为字母O
            'l': '1',  # 小写l可能被识别为数字1
            'S': '5',  # 字母S可能被识别为数字5
            'G': '6',  # 字母G可能被识别为数字6
            'B': '8',  # 字母B可能被识别为数字8
            'g': '9',  # 小写g可能被识别为数字9
            
            # 数学符号纠正
            'x': '×',  # 乘号
            '*': '×',  # 乘号
            '/': '÷',  # 除号
            '>=': '≥', # 大于等于
            '<=': '≤', # 小于等于
            '!=': '≠', # 不等于
            '+-': '±', # 正负号
        }
    
    def process_math_symbols(self, text: str) -> str:
        """处理数学符号"""
        processed_text = text
        
        # 识别并替换数学符号
        for symbol_type, variations in self.symbol_mappings.items():
            for variation in variations:
                if variation in processed_text:
                    # 根据符号类型进行标准化
                    standard_symbol = self._get_standard_symbol(symbol_type)
                    processed_text = processed_text.replace(variation, standard_symbol)
        
        # 应用纠错规则
        for incorrect, correct in self.correction_rules.items():
            processed_text = processed_text.replace(incorrect, correct)
            
        return processed_text
    
    def _get_standard_symbol(self, symbol_type: str) -> str:
        """获取标准符号"""
        standard_symbols = {
            'triangle': '△',
            'angle': '∠',
            'parallel': '∥',
            'perpendicular': '⊥',
            'degree': '°',
            'equal': '=',
            'plus': '+',
            'minus': '-',
        }
        return standard_symbols.get(symbol_type, symbol_type)
    
    def enhance_math_context(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """基于上下文增强数学符号识别"""
        enhanced_results = []
        
        for result in ocr_results:
            enhanced_text = self.process_math_symbols(result.text)
            
            # 提升数学符号的置信度
            if self._contains_math_symbols(enhanced_text):
                confidence_boost = 0.1
                new_confidence = min(1.0, result.confidence + confidence_boost)
            else:
                new_confidence = result.confidence
                
            enhanced_result = OCRResult(
                text=enhanced_text,
                confidence=new_confidence,
                bounding_box=result.bounding_box,
                engine=result.engine + "_enhanced"
            )
            enhanced_results.append(enhanced_result)
            
        return enhanced_results
    
    def _contains_math_symbols(self, text: str) -> bool:
        """检查是否包含数学符号"""
        math_symbols = ['△', '∠', '∥', '⊥', '°', '×', '÷', '±', '≥', '≤', '≠']
        return any(symbol in text for symbol in math_symbols)

class GeometryGuidedTextExtractor:
    """几何引导的文本提取器"""
    
    def __init__(self):
        self.config = OCR_CONFIG.TEXT_EXTRACTION
        
    def extract_text_regions(self, image: np.ndarray, 
                           geometry_mask: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """提取文本区域"""
        
        # 如果有几何掩码，则排除几何区域
        text_focused_image = self._apply_geometry_mask(image, geometry_mask) if geometry_mask is not None else image
        
        # 文本区域检测
        text_regions = self._detect_text_regions(text_focused_image)
        
        return text_regions
    
    def _apply_geometry_mask(self, image: np.ndarray, geometry_mask: np.ndarray) -> np.ndarray:
        """应用几何掩码，专注于非几何区域"""
        # 创建文本掩码（几何掩码的反转）
        text_mask = cv2.bitwise_not(geometry_mask)
        
        # 扩张文本区域以包含附近的文字
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)
        
        # 应用掩码
        result = cv2.bitwise_and(image, image, mask=text_mask)
        
        return result
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """检测文本区域"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用MSER(最稳定极值区域)检测文本
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        
        for region in regions:
            # 获取区域的边界框
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # 过滤太小或太大的区域
            if (w < self.config['min_text_height'] or 
                h < self.config['min_text_height'] or
                w > image.shape[1] * 0.8 or 
                h > self.config['max_text_height']):
                continue
            
            # 添加边距
            padding = self.config['text_padding']
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # 提取文本区域
            text_region = image[y:y+h, x:x+w]
            text_regions.append((text_region, (x, y, w, h)))
        
        return text_regions

class MathematicalOCR:
    """数学图表OCR主类"""
    
    def __init__(self, config=None):
        self.config = config if config else OCR_CONFIG
        self.engines = []
        self.symbol_processor = MathSymbolProcessor()
        self.text_extractor = GeometryGuidedTextExtractor()
        
        # 初始化可用的OCR引擎
        if EASYOCR_AVAILABLE:
            try:
                self.engines.append(EasyOCREngine())
            except Exception as e:
                logger.warning(f"EasyOCR初始化失败: {e}")
                
        if TESSERACT_AVAILABLE:
            try:
                self.engines.append(TesseractEngine())
            except Exception as e:
                logger.warning(f"Tesseract初始化失败: {e}")
        
        if not self.engines:
            logger.warning("没有可用的OCR引擎，文本提取功能将被禁用")
        else:
            logger.info(f"初始化了 {len(self.engines)} 个OCR引擎: {[e.name for e in self.engines]}")
    
    def extract_text(self, image: np.ndarray, 
                    geometry_elements: List[GeometryPrimitive] = None,
                    geometry_mask: Optional[np.ndarray] = None) -> List[TextElement]:
        """提取图像中的文本"""
        if not self.engines:
            logger.info("没有可用的OCR引擎，跳过文本提取")
            return []
            
        logger.info("开始数学OCR文本提取")
        
        # 几何引导的文本区域提取
        if geometry_mask is not None:
            text_regions = self.text_extractor.extract_text_regions(image, geometry_mask)
        else:
            text_regions = [(image, (0, 0, image.shape[1], image.shape[0]))]
        
        all_results = []
        
        # 对每个文本区域进行OCR
        for region_image, bbox in text_regions:
            region_results = self._extract_from_region(region_image, bbox)
            all_results.extend(region_results)
        
        # 多引擎融合
        fused_results = self._fuse_ocr_results(all_results)
        
        # 数学符号增强
        enhanced_results = self.symbol_processor.enhance_math_context(fused_results)
        
        # 转换为TextElement
        text_elements = self._convert_to_text_elements(enhanced_results, geometry_elements)
        
        logger.info(f"OCR完成，识别出 {len(text_elements)} 个文本元素")
        return text_elements
    
    def _extract_from_region(self, region_image: np.ndarray, 
                           bbox: Tuple[int, int, int, int]) -> List[OCRResult]:
        """从单个区域提取文本"""
        results = []
        x_offset, y_offset, _, _ = bbox
        
        for engine in self.engines:
            try:
                engine_results = engine.extract_text(region_image)
                
                # 调整边界框坐标（加上区域偏移）
                for result in engine_results:
                    min_point, max_point = result.bounding_box
                    adjusted_min = Point(min_point.x + x_offset, min_point.y + y_offset)
                    adjusted_max = Point(max_point.x + x_offset, max_point.y + y_offset)
                    
                    adjusted_result = OCRResult(
                        text=result.text,
                        confidence=result.confidence,
                        bounding_box=(adjusted_min, adjusted_max),
                        engine=result.engine
                    )
                    results.append(adjusted_result)
                    
            except Exception as e:
                logger.error(f"引擎 {engine.name} 处理区域失败: {e}")
        
        return results
    
    def _fuse_ocr_results(self, all_results: List[OCRResult]) -> List[OCRResult]:
        """融合多引擎OCR结果"""
        if not all_results:
            return []
        
        # 按空间位置聚类
        clusters = self._cluster_by_position(all_results)
        
        fused_results = []
        for cluster in clusters:
            fused_result = self._fuse_cluster(cluster)
            if fused_result:
                fused_results.append(fused_result)
                
        return fused_results
    
    def _cluster_by_position(self, results: List[OCRResult], 
                           distance_threshold: float = 20.0) -> List[List[OCRResult]]:
        """按位置聚类OCR结果"""
        if not results:
            return []
            
        clusters = []
        used = set()
        
        for i, result1 in enumerate(results):
            if i in used:
                continue
                
            cluster = [result1]
            used.add(i)
            center1 = result1.get_center()
            
            for j, result2 in enumerate(results):
                if j in used or i == j:
                    continue
                    
                center2 = result2.get_center()
                if center1.distance_to(center2) < distance_threshold:
                    cluster.append(result2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _fuse_cluster(self, cluster: List[OCRResult]) -> Optional[OCRResult]:
        """融合一个聚类的结果"""
        if not cluster:
            return None
            
        if len(cluster) == 1:
            return cluster[0]
        
        # 选择置信度最高的结果作为基础
        best_result = max(cluster, key=lambda r: r.confidence)
        
        # 如果有EasyOCR结果，优先选择
        easyocr_results = [r for r in cluster if 'easyocr' in r.engine]
        if easyocr_results:
            best_result = max(easyocr_results, key=lambda r: r.confidence)
        
        # 提升多引擎支持的置信度
        engine_count = len(set(r.engine for r in cluster))
        confidence_boost = min(0.2, engine_count * 0.05)
        final_confidence = min(1.0, best_result.confidence + confidence_boost)
        
        return OCRResult(
            text=best_result.text,
            confidence=final_confidence,
            bounding_box=best_result.bounding_box,
            engine=f"fused_{best_result.engine}"
        )
    
    def _convert_to_text_elements(self, ocr_results: List[OCRResult],
                                 geometry_elements: List[GeometryPrimitive] = None) -> List[TextElement]:
        """转换为TextElement并建立几何关联"""
        text_elements = []
        
        for result in ocr_results:
            min_point, max_point = result.bounding_box
            center = result.get_center()
            
            text_element = TextElement(
                text=result.text,
                position=center,
                bounding_box=result.bounding_box
            )
            text_element.confidence = result.confidence
            text_element.properties['ocr_engine'] = result.engine
            
            # 建立与几何元素的关联
            if geometry_elements:
                nearest_geometry = self._find_nearest_geometry(text_element, geometry_elements)
                if nearest_geometry:
                    text_element.properties['nearest_geometry'] = nearest_geometry.id
                    text_element.properties['distance_to_geometry'] = text_element.distance_to(nearest_geometry)
            
            text_elements.append(text_element)
        
        return text_elements
    
    def _find_nearest_geometry(self, text_element: TextElement, 
                              geometry_elements: List[GeometryPrimitive]) -> Optional[GeometryPrimitive]:
        """找到最近的几何元素"""
        if not geometry_elements:
            return None
            
        min_distance = float('inf')
        nearest_geometry = None
        
        for geometry in geometry_elements:
            distance = text_element.distance_to(geometry)
            if distance < min_distance:
                min_distance = distance
                nearest_geometry = geometry
        
        return nearest_geometry if min_distance < 100 else None  # 100像素阈值
    
    def extract_mathematical_text(self, image: np.ndarray, 
                                 geometry_elements: List[GeometryPrimitive] = None) -> List[Dict]:
        """提取数学文本的便捷方法，返回字典格式"""
        text_elements = self.extract_text(image, geometry_elements)
        
        # 转换为字典格式
        result = []
        for element in text_elements:
            text_dict = {
                'text': element.text,
                'position': element.position,
                'confidence': getattr(element, 'confidence', 0.0),
                'bounding_box': element.bounding_box,
                'properties': getattr(element, 'properties', {})
            }
            result.append(text_dict)
        
        return result

def create_geometry_mask(image_shape: Tuple[int, int], 
                        geometry_elements: List[GeometryPrimitive]) -> np.ndarray:
    """创建几何掩码"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for geometry in geometry_elements:
        if hasattr(geometry, 'vertices'):
            # 多边形
            points = np.array([(int(v.x), int(v.y)) for v in geometry.vertices], np.int32)
            cv2.fillPoly(mask, [points], 255)
        elif hasattr(geometry, 'center') and hasattr(geometry, 'radius'):
            # 圆形
            center = (int(geometry.center.x), int(geometry.center.y))
            cv2.circle(mask, center, int(geometry.radius), 255, -1)
        elif hasattr(geometry, 'start') and hasattr(geometry, 'end'):
            # 直线
            start = (int(geometry.start.x), int(geometry.start.y))
            end = (int(geometry.end.x), int(geometry.end.y))
            cv2.line(mask, start, end, 255, 5)  # 5像素粗线
    
    return mask

if __name__ == "__main__":
    # 测试代码
    print("测试数学OCR系统...")
    
    # 创建测试图像
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # 添加一些文本（模拟）
    cv2.putText(test_image, "Triangle ABC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "Angle = 90°", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "AB ∥ CD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    try:
        # 测试OCR系统
        ocr_system = MathematicalOCR()
        text_elements = ocr_system.extract_text(test_image)
        
        print(f"识别出 {len(text_elements)} 个文本元素:")
        for i, element in enumerate(text_elements):
            print(f"  {i+1}. '{element.text}' (置信度: {element.confidence:.3f})")
            print(f"     位置: {element.position}")
            print(f"     引擎: {element.properties.get('ocr_engine', 'unknown')}")
            print()
            
        # 测试数学符号处理
        symbol_processor = MathSymbolProcessor()
        test_texts = ["angle = 90 degree", "AB parallel CD", "triangle ABC"]
        
        print("数学符号处理测试:")
        for text in test_texts:
            processed = symbol_processor.process_math_symbols(text)
            print(f"  '{text}' -> '{processed}'")
            
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保安装了EasyOCR或Tesseract")