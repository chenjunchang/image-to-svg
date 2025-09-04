#!/usr/bin/env python3
"""
几何图元基类定义
定义系统中使用的各种几何图形的基础类和数据结构
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid

@dataclass
class Point:
    """二维点"""
    x: float
    y: float
    
    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一点的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def midpoint(self, other: 'Point') -> 'Point':
        """计算中点"""
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)
    
    def translate(self, dx: float, dy: float) -> 'Point':
        """平移点"""
        return Point(self.x + dx, self.y + dy)
    
    def rotate(self, center: 'Point', angle: float) -> 'Point':
        """绕指定中心旋转"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dx = self.x - center.x
        dy = self.y - center.y
        new_x = center.x + dx * cos_a - dy * sin_a
        new_y = center.y + dx * sin_a + dy * cos_a
        return Point(new_x, new_y)
    
    def to_tuple(self) -> Tuple[float, float]:
        """转换为元组"""
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        """转换为整数元组"""
        return (int(round(self.x)), int(round(self.y)))
    
    def __str__(self) -> str:
        return f"Point({self.x:.2f}, {self.y:.2f})"

class GeometryPrimitive(ABC):
    """几何图元基类"""
    
    def __init__(self, primitive_type: str):
        self.id = str(uuid.uuid4())
        self.type = primitive_type
        self.confidence = 1.0
        self.properties = {}
        self.detected_by = []  # 记录检测此图元的算法
        
    @abstractmethod
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """获取边界框"""
        pass
    
    @abstractmethod
    def get_center(self) -> Point:
        """获取几何中心"""
        pass
    
    @abstractmethod
    def get_area(self) -> float:
        """获取面积"""
        pass
    
    @abstractmethod
    def get_perimeter(self) -> float:
        """获取周长"""
        pass
    
    @abstractmethod
    def contains_point(self, point: Point) -> bool:
        """判断是否包含指定点"""
        pass
    
    @abstractmethod
    def distance_to_point(self, point: Point) -> float:
        """计算到点的距离"""
        pass
    
    def add_detection_source(self, detector_name: str, confidence: float):
        """添加检测来源"""
        self.detected_by.append({
            'detector': detector_name,
            'confidence': confidence
        })
        # 更新综合置信度（使用加权平均）
        if len(self.detected_by) == 1:
            self.confidence = confidence
        else:
            # 多检测器融合置信度
            total_confidence = sum(d['confidence'] for d in self.detected_by)
            self.confidence = min(1.0, total_confidence / len(self.detected_by) * 1.2)

class Line(GeometryPrimitive):
    """直线类"""
    
    def __init__(self, start: Point, end: Point):
        super().__init__("line")
        self.start = start
        self.end = end
        self.length = start.distance_to(end)
        self.angle = self._calculate_angle()
    
    def _calculate_angle(self) -> float:
        """计算直线与x轴的夹角"""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.arctan2(dy, dx)
    
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """获取边界框"""
        min_x = min(self.start.x, self.end.x)
        min_y = min(self.start.y, self.end.y)
        max_x = max(self.start.x, self.end.x)
        max_y = max(self.start.y, self.end.y)
        return Point(min_x, min_y), Point(max_x, max_y)
    
    def get_center(self) -> Point:
        """获取中点"""
        return self.start.midpoint(self.end)
    
    def get_area(self) -> float:
        """直线面积为0"""
        return 0.0
    
    def get_perimeter(self) -> float:
        """直线周长等于长度"""
        return self.length
    
    def contains_point(self, point: Point) -> bool:
        """判断点是否在直线上（考虑线段范围）"""
        return self.distance_to_point(point) < 1.0  # 1像素容差
    
    def distance_to_point(self, point: Point) -> float:
        """计算点到线段的距离"""
        A = self.end.y - self.start.y
        B = self.start.x - self.end.x
        C = self.end.x * self.start.y - self.start.x * self.end.y
        
        # 点到直线的距离
        line_distance = abs(A * point.x + B * point.y + C) / np.sqrt(A*A + B*B)
        
        # 检查点的投影是否在线段范围内
        dot = ((point.x - self.start.x) * (self.end.x - self.start.x) + 
               (point.y - self.start.y) * (self.end.y - self.start.y)) / (self.length * self.length)
        
        if dot < 0:
            return point.distance_to(self.start)
        elif dot > 1:
            return point.distance_to(self.end)
        else:
            return line_distance
    
    def is_parallel_to(self, other: 'Line', tolerance: float = 5.0) -> bool:
        """判断是否与另一条直线平行"""
        angle_diff = abs(self.angle - other.angle)
        return angle_diff < np.radians(tolerance) or angle_diff > np.pi - np.radians(tolerance)
    
    def is_perpendicular_to(self, other: 'Line', tolerance: float = 5.0) -> bool:
        """判断是否与另一条直线垂直"""
        angle_diff = abs(self.angle - other.angle)
        return abs(angle_diff - np.pi/2) < np.radians(tolerance) or abs(angle_diff - 3*np.pi/2) < np.radians(tolerance)

class Circle(GeometryPrimitive):
    """圆形类"""
    
    def __init__(self, center: Point, radius: float):
        super().__init__("circle")
        self.center = center
        self.radius = radius
    
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """获取边界框"""
        min_point = Point(self.center.x - self.radius, self.center.y - self.radius)
        max_point = Point(self.center.x + self.radius, self.center.y + self.radius)
        return min_point, max_point
    
    def get_center(self) -> Point:
        """获取圆心"""
        return self.center
    
    def get_area(self) -> float:
        """获取面积"""
        return np.pi * self.radius * self.radius
    
    def get_perimeter(self) -> float:
        """获取周长"""
        return 2 * np.pi * self.radius
    
    def contains_point(self, point: Point) -> bool:
        """判断点是否在圆内"""
        return self.center.distance_to(point) <= self.radius
    
    def distance_to_point(self, point: Point) -> float:
        """计算点到圆的距离"""
        center_distance = self.center.distance_to(point)
        return abs(center_distance - self.radius)

class Polygon(GeometryPrimitive):
    """多边形基类"""
    
    def __init__(self, vertices: List[Point], polygon_type: str = "polygon"):
        super().__init__(polygon_type)
        self.vertices = vertices
        self.edges = self._calculate_edges()
        self.angles = self._calculate_angles()
    
    def _calculate_edges(self) -> List[Line]:
        """计算边"""
        edges = []
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            edges.append(Line(start, end))
        return edges
    
    def _calculate_angles(self) -> List[float]:
        """计算内角"""
        angles = []
        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[(i - 1) % n]
            p2 = self.vertices[i]
            p3 = self.vertices[(i + 1) % n]
            
            # 计算两个向量的角度
            v1 = Point(p1.x - p2.x, p1.y - p2.y)
            v2 = Point(p3.x - p2.x, p3.y - p2.y)
            
            dot = v1.x * v2.x + v1.y * v2.y
            cross = v1.x * v2.y - v1.y * v2.x
            angle = np.arctan2(cross, dot)
            if angle < 0:
                angle += 2 * np.pi
                
            angles.append(angle)
        return angles
    
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """获取边界框"""
        x_coords = [v.x for v in self.vertices]
        y_coords = [v.y for v in self.vertices]
        min_point = Point(min(x_coords), min(y_coords))
        max_point = Point(max(x_coords), max(y_coords))
        return min_point, max_point
    
    def get_center(self) -> Point:
        """获取质心"""
        x_sum = sum(v.x for v in self.vertices)
        y_sum = sum(v.y for v in self.vertices)
        n = len(self.vertices)
        return Point(x_sum / n, y_sum / n)
    
    def get_area(self) -> float:
        """使用鞋带公式计算面积"""
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0
    
    def get_perimeter(self) -> float:
        """计算周长"""
        return sum(edge.length for edge in self.edges)
    
    def contains_point(self, point: Point) -> bool:
        """使用射线法判断点是否在多边形内"""
        x, y = point.x, point.y
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0].x, self.vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n].x, self.vertices[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def distance_to_point(self, point: Point) -> float:
        """计算点到多边形的距离"""
        if self.contains_point(point):
            return 0.0
        
        # 计算到各边的最小距离
        min_distance = float('inf')
        for edge in self.edges:
            distance = edge.distance_to_point(point)
            min_distance = min(min_distance, distance)
        
        return min_distance

class Triangle(Polygon):
    """三角形类"""
    
    def __init__(self, vertices: List[Point]):
        if len(vertices) != 3:
            raise ValueError("三角形必须有3个顶点")
        super().__init__(vertices, "triangle")
        self.side_lengths = [edge.length for edge in self.edges]
    
    def is_right_triangle(self, tolerance: float = 1.0) -> bool:
        """判断是否为直角三角形"""
        for angle in self.angles:
            if abs(angle - np.pi/2) < np.radians(tolerance):
                return True
        return False
    
    def is_equilateral(self, tolerance: float = 0.1) -> bool:
        """判断是否为等边三角形"""
        avg_length = sum(self.side_lengths) / 3
        for length in self.side_lengths:
            if abs(length - avg_length) > tolerance:
                return False
        return True
    
    def is_isosceles(self, tolerance: float = 0.1) -> bool:
        """判断是否为等腰三角形"""
        lengths = sorted(self.side_lengths)
        return (abs(lengths[0] - lengths[1]) < tolerance or 
                abs(lengths[1] - lengths[2]) < tolerance)

class Rectangle(Polygon):
    """矩形类"""
    
    def __init__(self, vertices: List[Point]):
        if len(vertices) != 4:
            raise ValueError("矩形必须有4个顶点")
        super().__init__(vertices, "rectangle")
        self.width, self.height = self._calculate_dimensions()
    
    def _calculate_dimensions(self) -> Tuple[float, float]:
        """计算宽度和高度"""
        # 假设顶点按顺序排列
        width = self.edges[0].length
        height = self.edges[1].length
        return width, height
    
    def is_square(self, tolerance: float = 0.1) -> bool:
        """判断是否为正方形"""
        return abs(self.width - self.height) < tolerance
    
    def get_aspect_ratio(self) -> float:
        """获取长宽比"""
        return max(self.width, self.height) / min(self.width, self.height)

class TextElement:
    """文本元素类"""
    
    def __init__(self, text: str, position: Point, bounding_box: Tuple[Point, Point]):
        self.id = str(uuid.uuid4())
        self.text = text
        self.position = position  # 文本位置
        self.bounding_box = bounding_box  # 边界框
        self.confidence = 1.0
        self.font_size = 12
        self.properties = {}
        
    def get_center(self) -> Point:
        """获取文本中心"""
        min_point, max_point = self.bounding_box
        return Point((min_point.x + max_point.x) / 2, 
                    (min_point.y + max_point.y) / 2)
    
    def distance_to(self, geometry: GeometryPrimitive) -> float:
        """计算到几何图元的距离"""
        text_center = self.get_center()
        return geometry.distance_to_point(text_center)
    
    def overlaps_with(self, geometry: GeometryPrimitive) -> bool:
        """判断是否与几何图元重叠"""
        min_point, max_point = self.bounding_box
        geometry_center = geometry.get_center()
        
        # 简单的边界框重叠检测
        return (min_point.x <= geometry_center.x <= max_point.x and 
                min_point.y <= geometry_center.y <= max_point.y)

# 工具函数
def create_geometry_from_contour(contour: np.ndarray, epsilon_factor: float = 0.02) -> Optional[GeometryPrimitive]:
    """从OpenCV轮廓创建几何图元"""
    # 多边形近似
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 3:
        return None
    
    # 转换为Point列表
    vertices = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    
    # 根据顶点数创建相应的几何图元
    if len(vertices) == 3:
        return Triangle(vertices)
    elif len(vertices) == 4:
        return Rectangle(vertices)
    else:
        return Polygon(vertices)

def create_line_from_hough(line_data: np.ndarray) -> Line:
    """从Hough变换结果创建直线"""
    x1, y1, x2, y2 = line_data
    start = Point(float(x1), float(y1))
    end = Point(float(x2), float(y2))
    return Line(start, end)

def create_circle_from_hough(circle_data: np.ndarray) -> Circle:
    """从Hough变换结果创建圆形"""
    x, y, r = circle_data
    center = Point(float(x), float(y))
    return Circle(center, float(r))

if __name__ == "__main__":
    # 测试代码
    print("几何图元测试:")
    
    # 创建点
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    print(f"点p1: {p1}")
    print(f"点p2: {p2}")
    print(f"距离: {p1.distance_to(p2)}")
    
    # 创建直线
    line = Line(p1, p2)
    print(f"\n直线: 长度={line.length:.2f}, 角度={np.degrees(line.angle):.1f}°")
    
    # 创建三角形
    p3 = Point(0, 3)
    triangle = Triangle([p1, p2, p3])
    print(f"\n三角形: 面积={triangle.get_area():.2f}, 周长={triangle.get_perimeter():.2f}")
    print(f"是否为直角三角形: {triangle.is_right_triangle()}")
    
    # 创建圆形
    circle = Circle(Point(0, 0), 5)
    print(f"\n圆形: 中心={circle.center}, 半径={circle.radius}")
    print(f"面积={circle.get_area():.2f}, 周长={circle.get_perimeter():.2f}")
    
    # 测试点与图形关系
    test_point = Point(1, 1)
    print(f"\n测试点 {test_point}:")
    print(f"在三角形内: {triangle.contains_point(test_point)}")
    print(f"在圆形内: {circle.contains_point(test_point)}")
    print(f"到直线距离: {line.distance_to_point(test_point):.2f}")