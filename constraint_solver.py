#!/usr/bin/env python3
"""
几何约束求解系统
实现CAD风格的几何约束识别、图构建和增量求解
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
import math

from geometry_primitives import GeometryPrimitive, Point, Line, Circle, Polygon, Triangle, Rectangle
from math_config import CONSTRAINT_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeometricRelationship:
    """几何关系"""
    elem1_id: str
    elem2_id: str
    relationship_type: str
    parameters: Dict[str, Any]
    confidence: float
    
    def __str__(self):
        return f"{self.relationship_type}({self.elem1_id}, {self.elem2_id}) - {self.confidence:.3f}"

class GeometricConstraint(ABC):
    """几何约束基类"""
    
    def __init__(self, constraint_type: str, elements: List[str], parameters: Dict[str, Any] = None):
        self.type = constraint_type
        self.elements = elements  # 涉及的几何元素ID列表
        self.parameters = parameters or {}
        self.tolerance = CONSTRAINT_CONFIG.CONSTRAINT_TOLERANCE.get(f'{constraint_type}_tolerance', 1.0)
        self.weight = CONSTRAINT_CONFIG.CONSTRAINT_WEIGHTS.get(constraint_type, 1.0)
        self.satisfied = False
        
    @abstractmethod
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算约束误差"""
        pass
    
    @abstractmethod
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用约束（修正几何元素位置）"""
        pass
    
    def is_satisfied(self, geometry_dict: Dict[str, GeometryPrimitive]) -> bool:
        """检查约束是否满足"""
        error = self.evaluate_error(geometry_dict)
        self.satisfied = error <= self.tolerance
        return self.satisfied

class ParallelConstraint(GeometricConstraint):
    """平行约束"""
    
    def __init__(self, line1_id: str, line2_id: str):
        super().__init__("parallel", [line1_id, line2_id])
        
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算平行约束误差"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return float('inf')
        
        # 计算角度差
        angle_diff = abs(line1.angle - line2.angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        # 转换为度数
        angle_error = np.degrees(min(angle_diff, np.pi - angle_diff))
        return angle_error
    
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用平行约束"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return geometry_dict
        
        # 计算平均角度
        avg_angle = (line1.angle + line2.angle) / 2
        
        # 调整两条直线的角度
        # 保持line1不变，调整line2
        length2 = line2.length
        center2 = line2.get_center()
        
        # 计算新的端点
        dx = length2 / 2 * np.cos(avg_angle)
        dy = length2 / 2 * np.sin(avg_angle)
        
        new_start = Point(center2.x - dx, center2.y - dy)
        new_end = Point(center2.x + dx, center2.y + dy)
        
        # 创建新的直线
        new_line2 = Line(new_start, new_end)
        new_line2.id = line2.id
        new_line2.confidence = line2.confidence
        
        # 更新几何字典
        updated_dict = geometry_dict.copy()
        updated_dict[self.elements[1]] = new_line2
        
        return updated_dict

class PerpendicularConstraint(GeometricConstraint):
    """垂直约束"""
    
    def __init__(self, line1_id: str, line2_id: str):
        super().__init__("perpendicular", [line1_id, line2_id])
        
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算垂直约束误差"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return float('inf')
        
        # 计算角度差
        angle_diff = abs(line1.angle - line2.angle)
        
        # 垂直应该是90度
        perpendicular_error = abs(angle_diff - np.pi/2)
        if perpendicular_error > np.pi/2:
            perpendicular_error = np.pi - perpendicular_error
            
        return np.degrees(perpendicular_error)
    
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用垂直约束"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return geometry_dict
        
        # 让line2垂直于line1
        perpendicular_angle = line1.angle + np.pi/2
        
        length2 = line2.length
        center2 = line2.get_center()
        
        # 计算新的端点
        dx = length2 / 2 * np.cos(perpendicular_angle)
        dy = length2 / 2 * np.sin(perpendicular_angle)
        
        new_start = Point(center2.x - dx, center2.y - dy)
        new_end = Point(center2.x + dx, center2.y + dy)
        
        new_line2 = Line(new_start, new_end)
        new_line2.id = line2.id
        new_line2.confidence = line2.confidence
        
        updated_dict = geometry_dict.copy()
        updated_dict[self.elements[1]] = new_line2
        
        return updated_dict

class IntersectionConstraint(GeometricConstraint):
    """相交约束"""
    
    def __init__(self, line1_id: str, line2_id: str, intersection_point: Optional[Point] = None):
        super().__init__("intersection", [line1_id, line2_id])
        self.intersection_point = intersection_point
        
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算相交约束误差"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return float('inf')
        
        # 计算直线交点
        intersection = self._calculate_intersection(line1, line2)
        if intersection is None:
            return float('inf')  # 平行线不相交
        
        if self.intersection_point:
            # 如果指定了交点位置，计算距离误差
            return self.intersection_point.distance_to(intersection)
        else:
            # 检查交点是否在两条线段上
            on_line1 = self._point_on_segment(intersection, line1)
            on_line2 = self._point_on_segment(intersection, line2)
            
            if on_line1 and on_line2:
                return 0.0
            else:
                # 计算延伸需要的距离
                dist1 = min(intersection.distance_to(line1.start), intersection.distance_to(line1.end))
                dist2 = min(intersection.distance_to(line2.start), intersection.distance_to(line2.end))
                return max(dist1, dist2)
    
    def _calculate_intersection(self, line1: Line, line2: Line) -> Optional[Point]:
        """计算两直线交点"""
        x1, y1 = line1.start.x, line1.start.y
        x2, y2 = line1.end.x, line1.end.y
        x3, y3 = line2.start.x, line2.start.y
        x4, y4 = line2.end.x, line2.end.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行或重合
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # 交点坐标
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return Point(px, py)
    
    def _point_on_segment(self, point: Point, line: Line, tolerance: float = 1.0) -> bool:
        """检查点是否在线段上"""
        return line.distance_to_point(point) < tolerance
    
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用相交约束"""
        # 对于相交约束，通常不需要大幅调整几何形状
        # 可以微调端点使其更精确地相交
        return geometry_dict

class DistanceConstraint(GeometricConstraint):
    """距离约束"""
    
    def __init__(self, elem1_id: str, elem2_id: str, target_distance: float):
        super().__init__("distance", [elem1_id, elem2_id])
        self.target_distance = target_distance
        
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算距离约束误差"""
        elem1 = geometry_dict[self.elements[0]]
        elem2 = geometry_dict[self.elements[1]]
        
        center1 = elem1.get_center()
        center2 = elem2.get_center()
        
        actual_distance = center1.distance_to(center2)
        return abs(actual_distance - self.target_distance)
    
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用距离约束"""
        # 距离约束的应用较复杂，这里提供简化版本
        return geometry_dict

class AngleConstraint(GeometricConstraint):
    """角度约束"""
    
    def __init__(self, line1_id: str, line2_id: str, target_angle: float):
        super().__init__("angle", [line1_id, line2_id])
        self.target_angle = target_angle  # 目标角度（弧度）
        
    def evaluate_error(self, geometry_dict: Dict[str, GeometryPrimitive]) -> float:
        """计算角度约束误差"""
        line1 = geometry_dict[self.elements[0]]
        line2 = geometry_dict[self.elements[1]]
        
        if not (isinstance(line1, Line) and isinstance(line2, Line)):
            return float('inf')
        
        angle_diff = abs(line1.angle - line2.angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
            
        actual_angle = min(angle_diff, np.pi - angle_diff)
        return abs(actual_angle - self.target_angle)
    
    def apply_constraint(self, geometry_dict: Dict[str, GeometryPrimitive]) -> Dict[str, GeometryPrimitive]:
        """应用角度约束"""
        # 角度约束的应用
        return geometry_dict

class RelationshipAnalyzer:
    """几何关系分析器"""
    
    def __init__(self):
        self.config = CONSTRAINT_CONFIG
        
    def analyze_relationships(self, geometry_elements: List[GeometryPrimitive]) -> List[GeometricRelationship]:
        """分析几何元素间的关系"""
        logger.info(f"分析 {len(geometry_elements)} 个几何元素的关系")
        
        relationships = []
        
        # 两两分析关系
        for i, elem1 in enumerate(geometry_elements):
            for j, elem2 in enumerate(geometry_elements[i+1:], i+1):
                relations = self._analyze_pair_relationship(elem1, elem2)
                relationships.extend(relations)
        
        logger.info(f"发现 {len(relationships)} 个几何关系")
        return relationships
    
    def _analyze_pair_relationship(self, elem1: GeometryPrimitive, elem2: GeometryPrimitive) -> List[GeometricRelationship]:
        """分析两个几何元素的关系"""
        relationships = []
        
        # 直线-直线关系
        if isinstance(elem1, Line) and isinstance(elem2, Line):
            # 平行关系
            if elem1.is_parallel_to(elem2, tolerance=10.0):
                confidence = self._calculate_parallel_confidence(elem1, elem2)
                relationships.append(GeometricRelationship(
                    elem1_id=elem1.id,
                    elem2_id=elem2.id,
                    relationship_type="parallel",
                    parameters={},
                    confidence=confidence
                ))
            
            # 垂直关系
            elif elem1.is_perpendicular_to(elem2, tolerance=10.0):
                confidence = self._calculate_perpendicular_confidence(elem1, elem2)
                relationships.append(GeometricRelationship(
                    elem1_id=elem1.id,
                    elem2_id=elem2.id,
                    relationship_type="perpendicular",
                    parameters={},
                    confidence=confidence
                ))
            
            # 相交关系
            intersection_point = self._find_intersection(elem1, elem2)
            if intersection_point:
                confidence = self._calculate_intersection_confidence(elem1, elem2, intersection_point)
                relationships.append(GeometricRelationship(
                    elem1_id=elem1.id,
                    elem2_id=elem2.id,
                    relationship_type="intersection",
                    parameters={"point": intersection_point},
                    confidence=confidence
                ))
        
        # 圆-直线关系
        elif isinstance(elem1, Circle) and isinstance(elem2, Line):
            tangent_confidence = self._analyze_circle_line_tangency(elem1, elem2)
            if tangent_confidence > 0.5:
                relationships.append(GeometricRelationship(
                    elem1_id=elem1.id,
                    elem2_id=elem2.id,
                    relationship_type="tangent",
                    parameters={},
                    confidence=tangent_confidence
                ))
        
        elif isinstance(elem1, Line) and isinstance(elem2, Circle):
            tangent_confidence = self._analyze_circle_line_tangency(elem2, elem1)
            if tangent_confidence > 0.5:
                relationships.append(GeometricRelationship(
                    elem1_id=elem1.id,
                    elem2_id=elem2.id,
                    relationship_type="tangent",
                    parameters={},
                    confidence=tangent_confidence
                ))
        
        # 多边形-直线关系（包含关系等）
        # TODO: 添加更多几何关系分析
        
        return relationships
    
    def _calculate_parallel_confidence(self, line1: Line, line2: Line) -> float:
        """计算平行关系的置信度"""
        angle_diff = abs(line1.angle - line2.angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        angle_error = np.degrees(min(angle_diff, np.pi - angle_diff))
        
        # 角度误差越小，置信度越高
        confidence = max(0.0, 1.0 - angle_error / 10.0)  # 10度容差
        return confidence
    
    def _calculate_perpendicular_confidence(self, line1: Line, line2: Line) -> float:
        """计算垂直关系的置信度"""
        angle_diff = abs(line1.angle - line2.angle)
        perpendicular_error = abs(angle_diff - np.pi/2)
        if perpendicular_error > np.pi/2:
            perpendicular_error = np.pi - perpendicular_error
        
        angle_error_degrees = np.degrees(perpendicular_error)
        confidence = max(0.0, 1.0 - angle_error_degrees / 10.0)
        return confidence
    
    def _find_intersection(self, line1: Line, line2: Line) -> Optional[Point]:
        """寻找两直线交点"""
        x1, y1 = line1.start.x, line1.start.y
        x2, y2 = line1.end.x, line1.end.y
        x3, y3 = line2.start.x, line2.start.y
        x4, y4 = line2.end.x, line2.end.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # 检查交点是否在线段范围内（扩展一点容差）
        if -0.1 <= t <= 1.1 and -0.1 <= u <= 1.1:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            return Point(px, py)
        
        return None
    
    def _calculate_intersection_confidence(self, line1: Line, line2: Line, intersection: Point) -> float:
        """计算相交关系的置信度"""
        # 基于交点到线段的距离计算置信度
        dist1 = line1.distance_to_point(intersection)
        dist2 = line2.distance_to_point(intersection)
        
        max_dist = max(dist1, dist2)
        confidence = max(0.0, 1.0 - max_dist / 20.0)  # 20像素容差
        return confidence
    
    def _analyze_circle_line_tangency(self, circle: Circle, line: Line) -> float:
        """分析圆-直线相切关系"""
        # 计算圆心到直线的距离
        distance = line.distance_to_point(circle.center)
        
        # 相切的条件是距离等于半径
        tangency_error = abs(distance - circle.radius)
        confidence = max(0.0, 1.0 - tangency_error / 5.0)  # 5像素容差
        
        return confidence

class IncrementalConstraintSolver:
    """增量约束求解器"""
    
    def __init__(self, config=None):
        self.config = config if config else CONSTRAINT_CONFIG
        self.constraint_graph = nx.Graph()
        self.constraints = []
        
    def solve_constraints(self, geometry_elements: List[GeometryPrimitive],
                         relationships: List[GeometricRelationship]) -> List[GeometryPrimitive]:
        """求解约束系统"""
        logger.info("开始增量约束求解")
        
        # 构建约束图
        self.constraint_graph = self._build_constraint_graph(geometry_elements, relationships)
        
        # 生成约束
        self.constraints = self._generate_constraints(relationships)
        
        # 按优先级分层求解
        solved_elements = self._solve_by_priority(geometry_elements)
        
        # 验证求解结果
        validation_result = self._validate_solution(solved_elements)
        
        logger.info(f"约束求解完成，验证结果: {validation_result}")
        return solved_elements
    
    def _build_constraint_graph(self, geometry_elements: List[GeometryPrimitive],
                               relationships: List[GeometricRelationship]) -> nx.Graph:
        """构建约束图"""
        graph = nx.Graph()
        
        # 添加几何元素作为节点
        for elem in geometry_elements:
            graph.add_node(elem.id, element=elem)
        
        # 添加关系作为边
        for rel in relationships:
            graph.add_edge(rel.elem1_id, rel.elem2_id, 
                          relationship=rel.relationship_type,
                          confidence=rel.confidence,
                          parameters=rel.parameters)
        
        return graph
    
    def _generate_constraints(self, relationships: List[GeometricRelationship]) -> List[GeometricConstraint]:
        """从关系生成约束"""
        constraints = []
        
        for rel in relationships:
            if rel.relationship_type == "parallel":
                constraints.append(ParallelConstraint(rel.elem1_id, rel.elem2_id))
            elif rel.relationship_type == "perpendicular":
                constraints.append(PerpendicularConstraint(rel.elem1_id, rel.elem2_id))
            elif rel.relationship_type == "intersection":
                intersection_point = rel.parameters.get("point")
                constraints.append(IntersectionConstraint(rel.elem1_id, rel.elem2_id, intersection_point))
            # 可以添加更多约束类型
        
        return constraints
    
    def _solve_by_priority(self, geometry_elements: List[GeometryPrimitive]) -> List[GeometryPrimitive]:
        """按优先级分层求解"""
        # 创建几何元素字典
        geometry_dict = {elem.id: elem for elem in geometry_elements}
        
        # 按优先级分组约束
        priority_groups = self._group_constraints_by_priority()
        
        for priority, constraints_group in priority_groups.items():
            logger.info(f"求解 {priority} 优先级约束 ({len(constraints_group)} 个)")
            
            # 对当前优先级的约束进行求解
            geometry_dict = self._solve_constraint_group(geometry_dict, constraints_group)
            
            # 验证当前优先级约束的满足情况
            satisfied_count = sum(1 for c in constraints_group if c.is_satisfied(geometry_dict))
            logger.info(f"优先级 {priority}: {satisfied_count}/{len(constraints_group)} 个约束满足")
        
        return list(geometry_dict.values())
    
    def _group_constraints_by_priority(self) -> Dict[str, List[GeometricConstraint]]:
        """按优先级分组约束"""
        priority_groups = {priority: [] for priority in self.config.CONSTRAINT_PRIORITY.keys()}
        
        for constraint in self.constraints:
            # 根据约束类型确定优先级
            assigned = False
            for priority, types in self.config.CONSTRAINT_PRIORITY.items():
                if constraint.type in types:
                    priority_groups[priority].append(constraint)
                    assigned = True
                    break
            
            if not assigned:
                priority_groups['optional'].append(constraint)
        
        return priority_groups
    
    def _solve_constraint_group(self, geometry_dict: Dict[str, GeometryPrimitive],
                               constraints: List[GeometricConstraint]) -> Dict[str, GeometryPrimitive]:
        """求解一组约束"""
        if not constraints:
            return geometry_dict
        
        max_iterations = 10
        convergence_threshold = 1e-3
        
        for iteration in range(max_iterations):
            previous_error = self._calculate_total_error(geometry_dict, constraints)
            
            # 对每个约束进行局部优化
            for constraint in constraints:
                if not constraint.is_satisfied(geometry_dict):
                    geometry_dict = constraint.apply_constraint(geometry_dict)
            
            # 检查收敛
            current_error = self._calculate_total_error(geometry_dict, constraints)
            
            if abs(previous_error - current_error) < convergence_threshold:
                logger.info(f"约束求解收敛，迭代次数: {iteration + 1}")
                break
                
        return geometry_dict
    
    def _calculate_total_error(self, geometry_dict: Dict[str, GeometryPrimitive],
                              constraints: List[GeometricConstraint]) -> float:
        """计算总约束误差"""
        total_error = 0.0
        for constraint in constraints:
            error = constraint.evaluate_error(geometry_dict)
            weighted_error = error * constraint.weight
            total_error += weighted_error * weighted_error
        
        return total_error
    
    def _validate_solution(self, solved_elements: List[GeometryPrimitive]) -> Dict[str, Any]:
        """验证求解结果"""
        geometry_dict = {elem.id: elem for elem in solved_elements}
        
        validation_results = {
            'total_constraints': len(self.constraints),
            'satisfied_constraints': 0,
            'constraint_satisfaction_rate': 0.0,
            'constraint_details': []
        }
        
        for constraint in self.constraints:
            is_satisfied = constraint.is_satisfied(geometry_dict)
            error = constraint.evaluate_error(geometry_dict)
            
            validation_results['constraint_details'].append({
                'type': constraint.type,
                'elements': constraint.elements,
                'satisfied': is_satisfied,
                'error': error,
                'tolerance': constraint.tolerance
            })
            
            if is_satisfied:
                validation_results['satisfied_constraints'] += 1
        
        if validation_results['total_constraints'] > 0:
            validation_results['constraint_satisfaction_rate'] = (
                validation_results['satisfied_constraints'] / validation_results['total_constraints']
            )
        
        return validation_results
    
    def get_constraint_count(self) -> int:
        """获取约束数量"""
        return len(self.constraints)

if __name__ == "__main__":
    # 测试代码
    print("测试约束求解系统...")
    
    # 创建测试几何元素
    line1 = Line(Point(0, 0), Point(100, 0))   # 水平线
    line1.id = "line1"
    
    line2 = Line(Point(0, 10), Point(100, 15))  # 接近平行的线
    line2.id = "line2"
    
    line3 = Line(Point(50, -50), Point(55, 50))  # 接近垂直的线
    line3.id = "line3"
    
    geometry_elements = [line1, line2, line3]
    
    # 分析几何关系
    print("分析几何关系...")
    analyzer = RelationshipAnalyzer()
    relationships = analyzer.analyze_relationships(geometry_elements)
    
    print(f"发现关系:")
    for rel in relationships:
        print(f"  {rel}")
    
    # 约束求解
    print("\n开始约束求解...")
    solver = IncrementalConstraintSolver()
    solved_elements = solver.solve_constraints(geometry_elements, relationships)
    
    print(f"求解完成，得到 {len(solved_elements)} 个几何元素")
    
    # 打印求解结果
    for elem in solved_elements:
        print(f"元素 {elem.id}: {elem.type}")
        if isinstance(elem, Line):
            print(f"  起点: {elem.start}, 终点: {elem.end}")
            print(f"  角度: {np.degrees(elem.angle):.1f}°")
        print()
    
    # 验证约束满足情况
    validation = solver._validate_solution(solved_elements)
    print(f"约束满足率: {validation['constraint_satisfaction_rate']:.2%}")
    print(f"满足约束: {validation['satisfied_constraints']}/{validation['total_constraints']}")
    
    for detail in validation['constraint_details']:
        status = "✓" if detail['satisfied'] else "✗"
        print(f"  {status} {detail['type']}: 误差={detail['error']:.2f}, 容差={detail['tolerance']:.2f}")