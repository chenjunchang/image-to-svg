#!/usr/bin/env python3
"""
几何图元检测器
实现混合几何检测策略，包含轮廓检测、Hough变换检测和模板匹配
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from abc import ABC, abstractmethod

from geometry_primitives import (
    GeometryPrimitive, Point, Line, Circle, Triangle, Rectangle, Polygon,
    create_geometry_from_contour, create_line_from_hough, create_circle_from_hough
)
from math_config import GEOMETRY_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometryDetector(ABC):
    """几何检测器抽象基类"""
    
    def __init__(self, detector_name: str):
        self.name = detector_name
        self.config = GEOMETRY_CONFIG
        
    @abstractmethod
    def detect_shapes(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """检测几何图形"""
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊减噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred

class ContourBasedDetector(GeometryDetector):
    """基于轮廓的几何检测器"""
    
    def __init__(self):
        super().__init__("contour_detector")
        
    def detect_shapes(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """使用轮廓检测几何图形"""
        logger.info(f"开始{self.name}检测")
        
        # 预处理
        gray = self.preprocess_image(image)
        
        # 边缘检测
        edges = cv2.Canny(gray, 
                         self.config.CANNY_PARAMS['low_threshold'],
                         self.config.CANNY_PARAMS['high_threshold'])
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # 过滤太小的轮廓
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if (area < self.config.CONTOUR_PARAMS['min_area'] or 
                perimeter < self.config.CONTOUR_PARAMS['min_perimeter']):
                continue
                
            # 多边形近似
            epsilon = self.config.CONTOUR_PARAMS['epsilon_factor'] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 创建几何图元
            shape = self._create_shape_from_approx(approx, area, perimeter)
            if shape:
                shape.add_detection_source(self.name, self._calculate_confidence(shape, contour))
                shapes.append(shape)
        
        logger.info(f"{self.name}检测到 {len(shapes)} 个图形")
        return shapes
    
    def _create_shape_from_approx(self, approx: np.ndarray, area: float, perimeter: float) -> Optional[GeometryPrimitive]:
        """从近似轮廓创建几何图形"""
        vertex_count = len(approx)
        
        # 转换为Point列表
        vertices = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        
        try:
            if vertex_count == 3:
                return Triangle(vertices)
            elif vertex_count == 4:
                # 判断是否为矩形
                shape = Rectangle(vertices) if self._is_rectangle_like(vertices) else Polygon(vertices)
                return shape
            elif vertex_count > 4 and vertex_count <= self.config.CONTOUR_PARAMS['max_vertices']:
                # 检查是否为圆形近似
                if self._is_circle_like(vertices, area, perimeter):
                    center = self._calculate_centroid(vertices)
                    radius = self._estimate_radius(vertices, center)
                    return Circle(center, radius)
                else:
                    return Polygon(vertices)
        except Exception as e:
            logger.warning(f"创建图形失败: {e}")
            return None
        
        return None
    
    def _is_rectangle_like(self, vertices: List[Point]) -> bool:
        """判断是否类似矩形"""
        if len(vertices) != 4:
            return False
            
        # 计算各边长度
        edges = []
        for i in range(4):
            start = vertices[i]
            end = vertices[(i + 1) % 4]
            edges.append(Line(start, end))
        
        # 检查对边是否相等（矩形特性）
        opposite_pairs = [(0, 2), (1, 3)]
        angle_tolerance = self.config.SHAPE_CLASSIFICATION['angle_tolerance']
        
        for pair in opposite_pairs:
            edge1, edge2 = edges[pair[0]], edges[pair[1]]
            length_diff = abs(edge1.length - edge2.length)
            if length_diff > edge1.length * 0.2:  # 20%容差
                return False
                
        # 检查相邻边是否垂直
        for i in range(4):
            edge1 = edges[i]
            edge2 = edges[(i + 1) % 4]
            if not edge1.is_perpendicular_to(edge2, angle_tolerance):
                return False
                
        return True
    
    def _is_circle_like(self, vertices: List[Point], area: float, perimeter: float) -> bool:
        """判断是否类似圆形"""
        if len(vertices) < 8:  # 圆形近似需要足够多的点
            return False
            
        # 计算理论圆形的面积和周长比值
        theoretical_ratio = 4 * np.pi
        actual_ratio = perimeter * perimeter / area if area > 0 else float('inf')
        
        # 圆形的周长²/面积 ≈ 4π
        ratio_error = abs(actual_ratio - theoretical_ratio) / theoretical_ratio
        return ratio_error < 0.15  # 15%容差
    
    def _calculate_centroid(self, vertices: List[Point]) -> Point:
        """计算质心"""
        x_sum = sum(v.x for v in vertices)
        y_sum = sum(v.y for v in vertices)
        n = len(vertices)
        return Point(x_sum / n, y_sum / n)
    
    def _estimate_radius(self, vertices: List[Point], center: Point) -> float:
        """估算半径"""
        distances = [center.distance_to(v) for v in vertices]
        return sum(distances) / len(distances)
    
    def _calculate_confidence(self, shape: GeometryPrimitive, contour: np.ndarray) -> float:
        """计算检测置信度"""
        # 基于轮廓完整性和规整度计算置信度
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.1
            
        # 紧密度 (4πA/P²)，圆形为1，其他形状小于1
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        
        # 基础置信度
        base_confidence = min(1.0, area / 10000)  # 基于面积的基础置信度
        
        # 形状特定的置信度调整
        if isinstance(shape, Circle):
            # 圆形：紧密度越接近1越好
            shape_confidence = 1 - abs(1 - compactness)
        elif isinstance(shape, Triangle):
            # 三角形：紧密度适中
            shape_confidence = 1 - abs(0.6 - compactness) if compactness < 0.6 else 0.8
        elif isinstance(shape, Rectangle):
            # 矩形：紧密度较低
            shape_confidence = 1 - abs(0.785 - compactness) if compactness < 0.785 else 0.9
        else:
            # 其他多边形
            shape_confidence = 0.7
            
        return min(1.0, (base_confidence + shape_confidence) / 2)

class HoughBasedDetector(GeometryDetector):
    """基于Hough变换的几何检测器"""
    
    def __init__(self):
        super().__init__("hough_detector")
        
    def detect_shapes(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """使用Hough变换检测几何图形"""
        logger.info(f"开始{self.name}检测")
        
        shapes = []
        gray = self.preprocess_image(image)
        edges = cv2.Canny(gray, 
                         self.config.CANNY_PARAMS['low_threshold'],
                         self.config.CANNY_PARAMS['high_threshold'])
        
        # 检测直线
        lines = self._detect_lines(edges)
        shapes.extend(lines)
        
        # 检测圆形
        circles = self._detect_circles(gray)
        shapes.extend(circles)
        
        logger.info(f"{self.name}检测到 {len(shapes)} 个图形")
        return shapes
    
    def _detect_lines(self, edges: np.ndarray) -> List[Line]:
        """检测直线"""
        lines_data = cv2.HoughLinesP(
            edges,
            rho=self.config.HOUGH_PARAMS['lines']['rho'],
            theta=self.config.HOUGH_PARAMS['lines']['theta'],
            threshold=self.config.HOUGH_PARAMS['lines']['threshold'],
            minLineLength=self.config.HOUGH_PARAMS['lines']['min_line_length'],
            maxLineGap=self.config.HOUGH_PARAMS['lines']['max_line_gap']
        )
        
        lines = []
        if lines_data is not None:
            # 合并相近的直线
            merged_lines = self._merge_similar_lines(lines_data)
            
            for line_data in merged_lines:
                line = create_line_from_hough(line_data)
                line.add_detection_source(self.name, self._calculate_line_confidence(line))
                lines.append(line)
        
        return lines
    
    def _detect_circles(self, gray: np.ndarray) -> List[Circle]:
        """检测圆形"""
        circles_data = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.config.HOUGH_PARAMS['circles']['dp'],
            minDist=self.config.HOUGH_PARAMS['circles']['min_dist'],
            param1=self.config.HOUGH_PARAMS['circles']['param1'],
            param2=self.config.HOUGH_PARAMS['circles']['param2'],
            minRadius=self.config.HOUGH_PARAMS['circles']['min_radius'],
            maxRadius=self.config.HOUGH_PARAMS['circles']['max_radius']
        )
        
        circles = []
        if circles_data is not None:
            circles_data = np.round(circles_data[0, :]).astype("int")
            for circle_data in circles_data:
                circle = create_circle_from_hough(circle_data)
                circle.add_detection_source(self.name, self._calculate_circle_confidence(circle))
                circles.append(circle)
        
        return circles
    
    def _merge_similar_lines(self, lines_data: np.ndarray, 
                           angle_threshold: float = 10.0, 
                           distance_threshold: float = 20.0) -> List[np.ndarray]:
        """合并相似的直线"""
        if len(lines_data) == 0:
            return []
            
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines_data):
            if i in used:
                continue
                
            x1, y1, x2, y2 = line1[0]
            line_obj1 = Line(Point(x1, y1), Point(x2, y2))
            
            group = [line1[0]]
            used.add(i)
            
            # 查找相似的直线
            for j, line2 in enumerate(lines_data):
                if j in used or i == j:
                    continue
                    
                x3, y3, x4, y4 = line2[0]
                line_obj2 = Line(Point(x3, y3), Point(x4, y4))
                
                # 检查角度相似性
                angle_diff = abs(line_obj1.angle - line_obj2.angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                    
                if np.degrees(angle_diff) < angle_threshold:
                    # 检查距离相似性（中点距离）
                    center1 = line_obj1.get_center()
                    center2 = line_obj2.get_center()
                    if center1.distance_to(center2) < distance_threshold:
                        group.append(line2[0])
                        used.add(j)
            
            # 合并组内的直线
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged_line = self._merge_line_group(group)
                merged.append(merged_line)
        
        return merged
    
    def _merge_line_group(self, line_group: List[np.ndarray]) -> np.ndarray:
        """合并一组直线"""
        # 找到所有端点
        points = []
        for line in line_group:
            x1, y1, x2, y2 = line
            points.extend([(x1, y1), (x2, y2)])
        
        # 使用PCA找到主方向
        points_array = np.array(points)
        mean = np.mean(points_array, axis=0)
        centered = points_array - mean
        
        # SVD分解
        _, _, V = np.linalg.svd(centered)
        direction = V[0]  # 主方向
        
        # 将所有点投影到主方向上
        projections = np.dot(centered, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # 重建端点
        start_point = mean + min_proj * direction
        end_point = mean + max_proj * direction
        
        return np.array([start_point[0], start_point[1], end_point[0], end_point[1]], dtype=int)
    
    def _calculate_line_confidence(self, line: Line) -> float:
        """计算直线检测置信度"""
        # 基于长度的置信度
        length_confidence = min(1.0, line.length / 100)  # 100像素为满分
        
        # Hough变换检测的直线通常比较准确
        base_confidence = 0.8
        
        return min(1.0, (base_confidence + length_confidence) / 2)
    
    def _calculate_circle_confidence(self, circle: Circle) -> float:
        """计算圆形检测置信度"""
        # 基于半径的置信度
        radius_confidence = min(1.0, circle.radius / 50)  # 50像素为满分
        
        # Hough圆检测通常比较准确
        base_confidence = 0.9
        
        return min(1.0, (base_confidence + radius_confidence) / 2)

class TemplateMatchingDetector(GeometryDetector):
    """基于模板匹配的几何检测器"""
    
    def __init__(self):
        super().__init__("template_detector")
        self._initialize_templates()
        
    def _initialize_templates(self):
        """初始化几何形状模板"""
        self.templates = {
            'triangle': self._create_triangle_templates(),
            'square': self._create_square_templates(),
            'circle': self._create_circle_templates()
        }
    
    def _create_triangle_templates(self) -> List[np.ndarray]:
        """创建三角形模板"""
        templates = []
        sizes = [20, 30, 40, 50]
        
        for size in sizes:
            # 等边三角形
            img = np.zeros((size*2, size*2), dtype=np.uint8)
            points = np.array([
                [size, size//4],
                [size//2, size*3//2],
                [size*3//2, size*3//2]
            ], np.int32)
            cv2.fillPoly(img, [points], 255)
            templates.append(img)
            
        return templates
    
    def _create_square_templates(self) -> List[np.ndarray]:
        """创建正方形模板"""
        templates = []
        sizes = [20, 30, 40, 50]
        
        for size in sizes:
            img = np.zeros((size*2, size*2), dtype=np.uint8)
            cv2.rectangle(img, (size//2, size//2), (size*3//2, size*3//2), 255, -1)
            templates.append(img)
            
        return templates
    
    def _create_circle_templates(self) -> List[np.ndarray]:
        """创建圆形模板"""
        templates = []
        radii = [10, 15, 20, 25]
        
        for radius in radii:
            img = np.zeros((radius*4, radius*4), dtype=np.uint8)
            cv2.circle(img, (radius*2, radius*2), radius, 255, -1)
            templates.append(img)
            
        return templates
    
    def detect_shapes(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """使用模板匹配检测几何图形"""
        logger.info(f"开始{self.name}检测")
        
        gray = self.preprocess_image(image)
        shapes = []
        
        # 对每种模板进行匹配
        for shape_type, templates in self.templates.items():
            for template in templates:
                matches = self._match_template(gray, template, shape_type)
                shapes.extend(matches)
        
        # 去除重复检测
        shapes = self._remove_duplicates(shapes)
        
        logger.info(f"{self.name}检测到 {len(shapes)} 个图形")
        return shapes
    
    def _match_template(self, image: np.ndarray, template: np.ndarray, 
                       shape_type: str, threshold: float = 0.7) -> List[GeometryPrimitive]:
        """模板匹配"""
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        shapes = []
        for pt in zip(*locations[::-1]):
            confidence = float(result[pt[1], pt[0]])
            
            # 根据模板大小和位置创建几何图元
            shape = self._create_shape_from_template(pt, template.shape, shape_type)
            if shape:
                shape.add_detection_source(self.name, confidence)
                shapes.append(shape)
                
        return shapes
    
    def _create_shape_from_template(self, location: Tuple[int, int], 
                                  template_shape: Tuple[int, int], 
                                  shape_type: str) -> Optional[GeometryPrimitive]:
        """从模板匹配结果创建几何图元"""
        x, y = location
        h, w = template_shape
        
        try:
            if shape_type == 'triangle':
                # 创建三角形
                vertices = [
                    Point(x + w//2, y + h//4),
                    Point(x + w//4, y + 3*h//4),
                    Point(x + 3*w//4, y + 3*h//4)
                ]
                return Triangle(vertices)
                
            elif shape_type == 'square':
                # 创建矩形
                vertices = [
                    Point(x + w//4, y + h//4),
                    Point(x + 3*w//4, y + h//4),
                    Point(x + 3*w//4, y + 3*h//4),
                    Point(x + w//4, y + 3*h//4)
                ]
                return Rectangle(vertices)
                
            elif shape_type == 'circle':
                # 创建圆形
                center = Point(x + w//2, y + h//2)
                radius = min(w, h) // 4
                return Circle(center, radius)
                
        except Exception as e:
            logger.warning(f"从模板创建图形失败: {e}")
            
        return None
    
    def _remove_duplicates(self, shapes: List[GeometryPrimitive], 
                          distance_threshold: float = 20.0) -> List[GeometryPrimitive]:
        """移除重复检测的图形"""
        if not shapes:
            return shapes
            
        unique_shapes = []
        for shape in shapes:
            is_duplicate = False
            shape_center = shape.get_center()
            
            for existing in unique_shapes:
                existing_center = existing.get_center()
                if (shape.type == existing.type and 
                    shape_center.distance_to(existing_center) < distance_threshold):
                    is_duplicate = True
                    # 保留置信度更高的
                    if shape.confidence > existing.confidence:
                        unique_shapes.remove(existing)
                        unique_shapes.append(shape)
                    break
                    
            if not is_duplicate:
                unique_shapes.append(shape)
                
        return unique_shapes

class HybridGeometryDetector:
    """混合几何检测器 - 融合多种检测算法"""
    
    def __init__(self, config=None):
        self.config = config if config else GEOMETRY_CONFIG
        self.detectors = [
            ContourBasedDetector(),
            HoughBasedDetector(),
            TemplateMatchingDetector()
        ]
        self.fusion_weights = {
            'contour_detector': 1.0,
            'hough_detector': 1.2,
            'template_detector': 0.8
        }
    
    def detect_shapes(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """融合多种检测器的结果"""
        logger.info("开始混合几何检测")
        
        all_detections = []
        
        # 收集所有检测器的结果
        for detector in self.detectors:
            try:
                shapes = detector.detect_shapes(image)
                # 应用权重调整
                weight = self.fusion_weights.get(detector.name, 1.0)
                for shape in shapes:
                    shape.confidence *= weight
                all_detections.extend(shapes)
            except Exception as e:
                logger.error(f"检测器 {detector.name} 出错: {e}")
        
        # 融合检测结果
        fused_shapes = self._consensus_fusion(all_detections)
        
        logger.info(f"混合检测完成，最终结果: {len(fused_shapes)} 个图形")
        return fused_shapes
    
    def _consensus_fusion(self, detections: List[GeometryPrimitive]) -> List[GeometryPrimitive]:
        """投票融合机制"""
        if not detections:
            return []
        
        # 按类型分组
        shape_groups = {}
        for shape in detections:
            key = shape.type
            if key not in shape_groups:
                shape_groups[key] = []
            shape_groups[key].append(shape)
        
        fused_results = []
        
        # 对每种类型的图形进行融合
        for shape_type, shapes in shape_groups.items():
            fused_shapes = self._fuse_similar_shapes(shapes)
            fused_results.extend(fused_shapes)
        
        return fused_results
    
    def detect_geometries(self, image: np.ndarray) -> List[GeometryPrimitive]:
        """检测几何图形的接口方法（适配转换器）"""
        return self.detect_shapes(image)
    
    def _fuse_similar_shapes(self, shapes: List[GeometryPrimitive], 
                           distance_threshold: float = 30.0) -> List[GeometryPrimitive]:
        """融合相似的图形"""
        if not shapes:
            return []
            
        clusters = []
        used = set()
        
        # 聚类相近的图形
        for i, shape1 in enumerate(shapes):
            if i in used:
                continue
                
            cluster = [shape1]
            used.add(i)
            center1 = shape1.get_center()
            
            for j, shape2 in enumerate(shapes):
                if j in used or i == j:
                    continue
                    
                center2 = shape2.get_center()
                if center1.distance_to(center2) < distance_threshold:
                    cluster.append(shape2)
                    used.add(j)
            
            clusters.append(cluster)
        
        # 每个聚类选择最佳候选
        fused_shapes = []
        for cluster in clusters:
            best_shape = self._select_best_candidate(cluster)
            fused_shapes.append(best_shape)
            
        return fused_shapes
    
    def _select_best_candidate(self, candidates: List[GeometryPrimitive]) -> GeometryPrimitive:
        """从候选中选择最佳图形"""
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算综合得分
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_fusion_score(candidate, candidates)
            scored_candidates.append((score, candidate))
        
        # 返回得分最高的
        try:
            scored_candidates.sort(reverse=True)
        except Exception as e:
            logger.error(f"排序错误: {e}")
            logger.error(f"scored_candidates内容: {[(type(score), type(candidate)) for score, candidate in scored_candidates]}")
            # 使用key参数排序
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_candidates[0][1]
        
        # 更新置信度为融合置信度
        detection_count = len([c for c in candidates if c.type == best_candidate.type])
        boost_factor = min(1.5, 1.0 + detection_count * 0.1)
        best_candidate.confidence = min(1.0, best_candidate.confidence * boost_factor)
        
        return best_candidate
    
    def _calculate_fusion_score(self, candidate: GeometryPrimitive, 
                               all_candidates: List[GeometryPrimitive]) -> float:
        """计算融合得分"""
        base_score = candidate.confidence
        
        # 多检测器支持加分
        detector_count = len(set(d['detector'] for d in candidate.detected_by))
        multi_detector_bonus = detector_count * 0.1
        
        # 几何质量加分（面积、规整度等）
        geometry_score = self._evaluate_geometry_quality(candidate)
        
        total_score = base_score + multi_detector_bonus + geometry_score
        return min(1.0, total_score)
    
    def _evaluate_geometry_quality(self, shape: GeometryPrimitive) -> float:
        """评估几何图形质量"""
        if isinstance(shape, Circle):
            # 圆形：半径合理性
            return 0.1 if 5 <= shape.radius <= 200 else 0.0
        elif isinstance(shape, Line):
            # 直线：长度合理性
            return 0.1 if 10 <= shape.length <= 500 else 0.0
        elif isinstance(shape, Polygon):
            # 多边形：面积和周长合理性
            area = shape.get_area()
            perimeter = shape.get_perimeter()
            return 0.1 if 100 <= area <= 50000 and 20 <= perimeter <= 1000 else 0.0
        
        return 0.0

if __name__ == "__main__":
    # 测试代码
    import os
    
    # 创建测试图像
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    test_image.fill(255)  # 白色背景
    
    # 绘制测试图形
    # 圆形
    cv2.circle(test_image, (100, 100), 40, (0, 0, 0), 2)
    
    # 矩形
    cv2.rectangle(test_image, (200, 50), (300, 150), (0, 0, 0), 2)
    
    # 三角形
    triangle_pts = np.array([[150, 200], [100, 300], [200, 300]], np.int32)
    cv2.polylines(test_image, [triangle_pts], True, (0, 0, 0), 2)
    
    # 直线
    cv2.line(test_image, (50, 350), (350, 350), (0, 0, 0), 2)
    
    # 测试检测器
    print("测试几何检测器...")
    
    # 测试单个检测器
    contour_detector = ContourBasedDetector()
    contour_results = contour_detector.detect_shapes(test_image)
    print(f"轮廓检测结果: {len(contour_results)} 个图形")
    
    hough_detector = HoughBasedDetector()
    hough_results = hough_detector.detect_shapes(test_image)
    print(f"Hough检测结果: {len(hough_results)} 个图形")
    
    # 测试混合检测器
    hybrid_detector = HybridGeometryDetector()
    final_results = hybrid_detector.detect_shapes(test_image)
    print(f"混合检测结果: {len(final_results)} 个图形")
    
    # 打印检测结果详情
    for i, shape in enumerate(final_results):
        print(f"图形 {i+1}: {shape.type}, 置信度: {shape.confidence:.3f}")
        print(f"  检测来源: {[d['detector'] for d in shape.detected_by]}")
        print(f"  中心位置: {shape.get_center()}")
        if hasattr(shape, 'get_area'):
            print(f"  面积: {shape.get_area():.2f}")
        print()