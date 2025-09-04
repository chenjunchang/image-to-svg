#!/usr/bin/env python3
"""
数学语义分析器
理解数学图表的语义结构，建立文字-几何元素关联，推断数学含义
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
import re
from collections import defaultdict

from geometry_primitives import GeometryPrimitive, TextElement, Point, Line, Circle, Triangle, Rectangle, Polygon
from constraint_solver import GeometricRelationship
from math_config import OCR_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeometryTextBinding:
    """几何-文本绑定"""
    text: TextElement
    geometry: GeometryPrimitive
    relationship_type: str  # 'label', 'measurement', 'annotation', 'formula'
    confidence: float
    spatial_relationship: str  # 'inside', 'near', 'above', 'below', 'left', 'right'
    
    def __str__(self):
        return f"'{self.text.text}' -> {self.geometry.type}({self.geometry.id}) [{self.relationship_type}]"

@dataclass
class MathematicalConcept:
    """数学概念"""
    concept_type: str  # 'triangle', 'angle', 'parallel', 'perpendicular', etc.
    elements: List[str]  # 涉及的几何元素ID
    properties: Dict[str, Any]
    confidence: float
    description: str

@dataclass
class MathematicalInterpretation:
    """数学解释"""
    concept: MathematicalConcept
    supporting_evidence: List[str]  # 支持证据（文本、几何关系等）
    mathematical_meaning: str
    confidence: float

class MathematicalPatternMatcher:
    """数学模式匹配器"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """初始化数学模式"""
        return {
            'triangle_patterns': [
                {
                    'text_pattern': r'三角形\s*([A-Z]{3})',
                    'geometry_requirement': 'triangle',
                    'concept': 'triangle_naming'
                },
                {
                    'text_pattern': r'△\s*([A-Z]{3})',
                    'geometry_requirement': 'triangle',
                    'concept': 'triangle_symbol'
                }
            ],
            'angle_patterns': [
                {
                    'text_pattern': r'∠\s*([A-Z]{3})',
                    'geometry_requirement': 'angle',
                    'concept': 'angle_notation'
                },
                {
                    'text_pattern': r'角\s*([A-Z]+)',
                    'geometry_requirement': 'angle',
                    'concept': 'angle_chinese'
                },
                {
                    'text_pattern': r'(\d+)°',
                    'geometry_requirement': 'angle',
                    'concept': 'angle_measurement'
                }
            ],
            'parallel_patterns': [
                {
                    'text_pattern': r'([A-Z]+)\s*∥\s*([A-Z]+)',
                    'geometry_requirement': 'parallel_lines',
                    'concept': 'parallel_relation'
                },
                {
                    'text_pattern': r'([A-Z]+)\s*平行\s*([A-Z]+)',
                    'geometry_requirement': 'parallel_lines',
                    'concept': 'parallel_chinese'
                }
            ],
            'perpendicular_patterns': [
                {
                    'text_pattern': r'([A-Z]+)\s*⊥\s*([A-Z]+)',
                    'geometry_requirement': 'perpendicular_lines',
                    'concept': 'perpendicular_relation'
                },
                {
                    'text_pattern': r'([A-Z]+)\s*垂直\s*([A-Z]+)',
                    'geometry_requirement': 'perpendicular_lines',
                    'concept': 'perpendicular_chinese'
                }
            ],
            'measurement_patterns': [
                {
                    'text_pattern': r'([A-Z]+)\s*=\s*(\d+(?:\.\d+)?)',
                    'geometry_requirement': 'any',
                    'concept': 'length_measurement'
                },
                {
                    'text_pattern': r'面积\s*=\s*(\d+(?:\.\d+)?)',
                    'geometry_requirement': 'polygon',
                    'concept': 'area_measurement'
                }
            ]
        }
    
    def match_patterns(self, text: str, nearby_geometry: List[GeometryPrimitive]) -> List[Dict[str, Any]]:
        """匹配数学模式"""
        matches = []
        
        for pattern_category, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['text_pattern']
                requirement = pattern_info['geometry_requirement']
                concept = pattern_info['concept']
                
                # 文本模式匹配
                regex_matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in regex_matches:
                    # 检查几何要求
                    if self._check_geometry_requirement(nearby_geometry, requirement):
                        matches.append({
                            'concept': concept,
                            'match': match,
                            'groups': match.groups(),
                            'geometry': nearby_geometry,
                            'confidence': self._calculate_pattern_confidence(match, nearby_geometry)
                        })
        
        return matches
    
    def _check_geometry_requirement(self, geometry_list: List[GeometryPrimitive], requirement: str) -> bool:
        """检查几何要求是否满足"""
        if requirement == 'any':
            return True
        elif requirement == 'triangle':
            return any(isinstance(g, Triangle) for g in geometry_list)
        elif requirement == 'angle':
            # 简化：有两条直线就可能形成角
            lines = [g for g in geometry_list if isinstance(g, Line)]
            return len(lines) >= 2
        elif requirement == 'parallel_lines':
            lines = [g for g in geometry_list if isinstance(g, Line)]
            return len(lines) >= 2
        elif requirement == 'perpendicular_lines':
            lines = [g for g in geometry_list if isinstance(g, Line)]
            return len(lines) >= 2
        elif requirement == 'polygon':
            return any(isinstance(g, (Triangle, Rectangle, Polygon)) for g in geometry_list)
        
        return False
    
    def _calculate_pattern_confidence(self, match, geometry_list: List[GeometryPrimitive]) -> float:
        """计算模式匹配置信度"""
        base_confidence = 0.7  # 基础置信度
        
        # 根据几何元素数量调整
        geometry_bonus = min(0.2, len(geometry_list) * 0.05)
        
        # 根据匹配精度调整
        match_precision = len(match.group(0)) / 50.0  # 匹配长度贡献
        match_precision = min(0.1, match_precision)
        
        return min(1.0, base_confidence + geometry_bonus + match_precision)

class SceneGraphBuilder:
    """场景图构建器"""
    
    def __init__(self):
        self.pattern_matcher = MathematicalPatternMatcher()
    
    def build_scene_graph(self, geometry_elements: List[GeometryPrimitive],
                         text_elements: List[TextElement],
                         relationships: List[GeometricRelationship]) -> nx.DiGraph:
        """构建几何场景图"""
        logger.info("构建数学场景图")
        
        scene_graph = nx.DiGraph()
        
        # 添加几何元素节点
        for elem in geometry_elements:
            scene_graph.add_node(elem.id, 
                               type='geometry', 
                               element=elem, 
                               geometry_type=elem.type)
        
        # 添加文本元素节点
        for text in text_elements:
            scene_graph.add_node(text.id, 
                               type='text', 
                               element=text, 
                               content=text.text)
        
        # 添加几何关系边
        for rel in relationships:
            scene_graph.add_edge(rel.elem1_id, rel.elem2_id,
                                type='geometric_relation',
                                relation=rel.relationship_type,
                                confidence=rel.confidence)
        
        # 建立空间邻近关系
        self._add_spatial_relationships(scene_graph, geometry_elements, text_elements)
        
        # 添加数学语义边
        self._add_semantic_relationships(scene_graph, geometry_elements, text_elements)
        
        logger.info(f"场景图构建完成: {scene_graph.number_of_nodes()} 节点, {scene_graph.number_of_edges()} 边")
        return scene_graph
    
    def _add_spatial_relationships(self, graph: nx.DiGraph,
                                  geometry_elements: List[GeometryPrimitive],
                                  text_elements: List[TextElement]):
        """添加空间邻近关系"""
        proximity_threshold = 50.0  # 50像素邻近阈值
        
        for text in text_elements:
            text_center = text.get_center()
            
            # 找到最近的几何元素
            nearest_geometries = []
            for geometry in geometry_elements:
                distance = geometry.distance_to_point(text_center)
                if distance < proximity_threshold:
                    nearest_geometries.append((geometry, distance))
            
            # 按距离排序
            nearest_geometries.sort(key=lambda x: x[1])
            
            # 添加空间关系边
            for geometry, distance in nearest_geometries[:3]:  # 最多3个最近的
                spatial_rel = self._determine_spatial_relationship(text, geometry)
                confidence = max(0.1, 1.0 - distance / proximity_threshold)
                
                graph.add_edge(text.id, geometry.id,
                             type='spatial_relation',
                             relation=spatial_rel,
                             distance=distance,
                             confidence=confidence)
    
    def _determine_spatial_relationship(self, text: TextElement, geometry: GeometryPrimitive) -> str:
        """确定空间关系"""
        text_center = text.get_center()
        geometry_center = geometry.get_center()
        
        # 计算相对位置
        dx = text_center.x - geometry_center.x
        dy = text_center.y - geometry_center.y
        
        # 判断主要方向
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'below' if dy > 0 else 'above'
    
    def _add_semantic_relationships(self, graph: nx.DiGraph,
                                  geometry_elements: List[GeometryPrimitive],
                                  text_elements: List[TextElement]):
        """添加语义关系"""
        for text in text_elements:
            # 获取附近的几何元素
            nearby_geometries = self._get_nearby_geometries(graph, text.id, geometry_elements)
            
            # 模式匹配
            pattern_matches = self.pattern_matcher.match_patterns(text.text, nearby_geometries)
            
            # 添加语义边
            for match_info in pattern_matches:
                concept = match_info['concept']
                confidence = match_info['confidence']
                
                for geometry in match_info['geometry']:
                    if graph.has_edge(text.id, geometry.id):
                        # 更新现有边的语义信息
                        edge_data = graph.get_edge_data(text.id, geometry.id)
                        edge_data['semantic_concept'] = concept
                        edge_data['semantic_confidence'] = confidence
                    else:
                        # 添加新的语义边
                        graph.add_edge(text.id, geometry.id,
                                     type='semantic_relation',
                                     concept=concept,
                                     confidence=confidence)
    
    def _get_nearby_geometries(self, graph: nx.DiGraph, text_id: str,
                              geometry_elements: List[GeometryPrimitive]) -> List[GeometryPrimitive]:
        """获取附近的几何元素"""
        nearby_geometries = []
        
        # 从图中获取与文本相关的几何元素
        for neighbor in graph.neighbors(text_id):
            edge_data = graph.get_edge_data(text_id, neighbor)
            if edge_data and edge_data.get('type') == 'spatial_relation':
                # 找到对应的几何元素
                for geometry in geometry_elements:
                    if geometry.id == neighbor:
                        nearby_geometries.append(geometry)
                        break
        
        return nearby_geometries

class MathSemanticAnalyzer:
    """数学语义分析器主类"""
    
    def __init__(self, config=None):
        self.config = config
        self.scene_graph_builder = SceneGraphBuilder()
        self.math_concepts = OCR_CONFIG.MATH_SYMBOLS
        
    def analyze_mathematical_semantics(self, geometry_elements: List[GeometryPrimitive],
                                     text_elements: List[TextElement],
                                     relationships: List[GeometricRelationship]) -> Dict[str, Any]:
        """分析数学语义"""
        logger.info("开始数学语义分析")
        
        # 构建场景图
        scene_graph = self.scene_graph_builder.build_scene_graph(
            geometry_elements, text_elements, relationships
        )
        
        # 建立文字-几何绑定
        bindings = self._create_geometry_text_bindings(scene_graph, geometry_elements, text_elements)
        
        # 推断数学含义
        interpretations = self._infer_mathematical_meanings(scene_graph, bindings)
        
        # 建立数学坐标系
        coordinate_system = self._establish_coordinate_system(geometry_elements, text_elements)
        
        # 整合结果
        semantic_analysis = {
            'scene_graph': scene_graph,
            'text_geometry_bindings': bindings,
            'mathematical_interpretations': interpretations,
            'coordinate_system': coordinate_system,
            'confidence_metrics': self._calculate_confidence_metrics(bindings, interpretations)
        }
        
        logger.info(f"语义分析完成: {len(bindings)} 个绑定, {len(interpretations)} 个解释")
        return semantic_analysis
    
    def _create_geometry_text_bindings(self, scene_graph: nx.DiGraph,
                                     geometry_elements: List[GeometryPrimitive],
                                     text_elements: List[TextElement]) -> List[GeometryTextBinding]:
        """创建几何-文本绑定"""
        bindings = []
        
        for text in text_elements:
            # 获取与文本相关的几何元素
            related_geometries = []
            
            for neighbor in scene_graph.neighbors(text.id):
                edge_data = scene_graph.get_edge_data(text.id, neighbor)
                if edge_data and edge_data.get('type') in ['spatial_relation', 'semantic_relation']:
                    # 找到几何元素
                    geometry = self._find_geometry_by_id(geometry_elements, neighbor)
                    if geometry:
                        related_geometries.append((geometry, edge_data))
            
            # 为每个相关几何元素创建绑定
            for geometry, edge_data in related_geometries:
                relationship_type = self._determine_binding_type(text, geometry, edge_data)
                spatial_relationship = edge_data.get('relation', 'near')
                confidence = edge_data.get('confidence', 0.5)
                
                # 根据语义概念调整置信度
                if 'semantic_concept' in edge_data:
                    confidence = max(confidence, edge_data.get('semantic_confidence', 0.5))
                
                binding = GeometryTextBinding(
                    text=text,
                    geometry=geometry,
                    relationship_type=relationship_type,
                    confidence=confidence,
                    spatial_relationship=spatial_relationship
                )
                bindings.append(binding)
        
        return bindings
    
    def _find_geometry_by_id(self, geometry_elements: List[GeometryPrimitive], element_id: str) -> Optional[GeometryPrimitive]:
        """根据ID查找几何元素"""
        for geometry in geometry_elements:
            if geometry.id == element_id:
                return geometry
        return None
    
    def _determine_binding_type(self, text: TextElement, geometry: GeometryPrimitive, edge_data: Dict) -> str:
        """确定绑定类型"""
        text_content = text.text.lower()
        
        # 检查是否有语义概念
        if 'semantic_concept' in edge_data:
            concept = edge_data['semantic_concept']
            if 'measurement' in concept:
                return 'measurement'
            elif 'naming' in concept or 'notation' in concept:
                return 'label'
            elif 'relation' in concept:
                return 'annotation'
        
        # 基于文本内容判断
        if re.search(r'\d+(\.\d+)?\s*(cm|mm|°)', text_content):
            return 'measurement'
        elif re.search(r'^[A-Z]{1,3}$', text.text):  # 单个或几个大写字母
            return 'label'
        elif any(symbol in text_content for symbol in ['=', '∠', '△', '∥', '⊥']):
            return 'formula'
        else:
            return 'annotation'
    
    def _infer_mathematical_meanings(self, scene_graph: nx.DiGraph,
                                   bindings: List[GeometryTextBinding]) -> List[MathematicalInterpretation]:
        """推断数学含义"""
        interpretations = []
        
        # 按几何元素分组绑定
        geometry_bindings = defaultdict(list)
        for binding in bindings:
            geometry_bindings[binding.geometry.id].append(binding)
        
        # 为每个几何元素推断含义
        for geometry_id, related_bindings in geometry_bindings.items():
            geometry = related_bindings[0].geometry  # 获取几何元素
            
            # 分析三角形语义
            if isinstance(geometry, Triangle):
                triangle_interpretations = self._analyze_triangle_semantics(geometry, related_bindings, scene_graph)
                interpretations.extend(triangle_interpretations)
            
            # 分析直线语义
            elif isinstance(geometry, Line):
                line_interpretations = self._analyze_line_semantics(geometry, related_bindings, scene_graph)
                interpretations.extend(line_interpretations)
            
            # 分析圆形语义
            elif isinstance(geometry, Circle):
                circle_interpretations = self._analyze_circle_semantics(geometry, related_bindings, scene_graph)
                interpretations.extend(circle_interpretations)
        
        return interpretations
    
    def _analyze_triangle_semantics(self, triangle: Triangle, bindings: List[GeometryTextBinding],
                                  scene_graph: nx.DiGraph) -> List[MathematicalInterpretation]:
        """分析三角形语义"""
        interpretations = []
        
        # 查找三角形命名
        naming_bindings = [b for b in bindings if b.relationship_type == 'label']
        if naming_bindings:
            # 创建三角形命名概念
            concept = MathematicalConcept(
                concept_type='triangle_naming',
                elements=[triangle.id],
                properties={'name': naming_bindings[0].text.text},
                confidence=naming_bindings[0].confidence,
                description=f"三角形命名为 {naming_bindings[0].text.text}"
            )
            
            interpretation = MathematicalInterpretation(
                concept=concept,
                supporting_evidence=[b.text.text for b in naming_bindings],
                mathematical_meaning=f"这是一个名为'{concept.properties['name']}'的三角形",
                confidence=concept.confidence
            )
            interpretations.append(interpretation)
        
        # 分析三角形性质
        if triangle.is_right_triangle():
            concept = MathematicalConcept(
                concept_type='right_triangle',
                elements=[triangle.id],
                properties={'type': 'right'},
                confidence=0.9,
                description="这是一个直角三角形"
            )
            
            interpretation = MathematicalInterpretation(
                concept=concept,
                supporting_evidence=['几何分析'],
                mathematical_meaning="这是一个直角三角形，有一个90度角",
                confidence=0.9
            )
            interpretations.append(interpretation)
        
        return interpretations
    
    def _analyze_line_semantics(self, line: Line, bindings: List[GeometryTextBinding],
                              scene_graph: nx.DiGraph) -> List[MathematicalInterpretation]:
        """分析直线语义"""
        interpretations = []
        
        # 查找直线标签
        label_bindings = [b for b in bindings if b.relationship_type == 'label']
        if label_bindings:
            concept = MathematicalConcept(
                concept_type='line_naming',
                elements=[line.id],
                properties={'name': label_bindings[0].text.text},
                confidence=label_bindings[0].confidence,
                description=f"直线标记为 {label_bindings[0].text.text}"
            )
            
            interpretation = MathematicalInterpretation(
                concept=concept,
                supporting_evidence=[b.text.text for b in label_bindings],
                mathematical_meaning=f"这是直线'{concept.properties['name']}'",
                confidence=concept.confidence
            )
            interpretations.append(interpretation)
        
        return interpretations
    
    def _analyze_circle_semantics(self, circle: Circle, bindings: List[GeometryTextBinding],
                                scene_graph: nx.DiGraph) -> List[MathematicalInterpretation]:
        """分析圆形语义"""
        interpretations = []
        
        # 查找半径或直径标注
        measurement_bindings = [b for b in bindings if b.relationship_type == 'measurement']
        for binding in measurement_bindings:
            text = binding.text.text
            
            # 提取数值
            number_match = re.search(r'(\d+(?:\.\d+)?)', text)
            if number_match:
                value = float(number_match.group(1))
                
                if 'r' in text.lower() or '半径' in text:
                    concept = MathematicalConcept(
                        concept_type='circle_radius',
                        elements=[circle.id],
                        properties={'radius': value},
                        confidence=binding.confidence,
                        description=f"圆的半径为 {value}"
                    )
                elif 'd' in text.lower() or '直径' in text:
                    concept = MathematicalConcept(
                        concept_type='circle_diameter',
                        elements=[circle.id],
                        properties={'diameter': value},
                        confidence=binding.confidence,
                        description=f"圆的直径为 {value}"
                    )
                else:
                    continue
                
                interpretation = MathematicalInterpretation(
                    concept=concept,
                    supporting_evidence=[text],
                    mathematical_meaning=concept.description,
                    confidence=concept.confidence
                )
                interpretations.append(interpretation)
        
        return interpretations
    
    def _establish_coordinate_system(self, geometry_elements: List[GeometryPrimitive],
                                   text_elements: List[TextElement]) -> Dict[str, Any]:
        """建立数学坐标系"""
        # 计算图形边界
        all_points = []
        for elem in geometry_elements:
            if hasattr(elem, 'vertices'):
                all_points.extend([(v.x, v.y) for v in elem.vertices])
            elif hasattr(elem, 'center'):
                all_points.append((elem.center.x, elem.center.y))
                if hasattr(elem, 'radius'):
                    # 圆形的边界点
                    all_points.extend([
                        (elem.center.x - elem.radius, elem.center.y),
                        (elem.center.x + elem.radius, elem.center.y),
                        (elem.center.x, elem.center.y - elem.radius),
                        (elem.center.x, elem.center.y + elem.radius)
                    ])
            elif hasattr(elem, 'start') and hasattr(elem, 'end'):
                all_points.extend([(elem.start.x, elem.start.y), (elem.end.x, elem.end.y)])
        
        # 添加文本位置
        for text in text_elements:
            all_points.append((text.position.x, text.position.y))
        
        if not all_points:
            return {}
        
        # 计算边界
        x_coords, y_coords = zip(*all_points)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 建立坐标系
        coordinate_system = {
            'origin': Point(min_x, min_y),
            'bounds': {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            'center': Point((min_x + max_x) / 2, (min_y + max_y) / 2),
            'scale': max(max_x - min_x, max_y - min_y) / 100.0  # 归一化比例
        }
        
        return coordinate_system
    
    def _calculate_confidence_metrics(self, bindings: List[GeometryTextBinding],
                                    interpretations: List[MathematicalInterpretation]) -> Dict[str, float]:
        """计算置信度指标"""
        if not bindings:
            return {'average_binding_confidence': 0.0, 'average_interpretation_confidence': 0.0}
        
        # 计算平均绑定置信度
        avg_binding_confidence = sum(b.confidence for b in bindings) / len(bindings)
        
        # 计算平均解释置信度
        if interpretations:
            avg_interpretation_confidence = sum(i.confidence for i in interpretations) / len(interpretations)
        else:
            avg_interpretation_confidence = 0.0
        
        return {
            'average_binding_confidence': avg_binding_confidence,
            'average_interpretation_confidence': avg_interpretation_confidence,
            'total_bindings': len(bindings),
            'total_interpretations': len(interpretations)
        }
    
    def analyze_mathematical_scene(self, geometry_elements: List[GeometryPrimitive],
                                  text_elements: List[Dict], image: np.ndarray) -> Dict[str, Any]:
        """分析数学场景，适配转换器接口"""
        # 转换文本格式
        text_element_objects = []
        for text_dict in text_elements:
            text_obj = TextElement(
                text=text_dict['text'],
                position=text_dict['position'],
                bounding_box=text_dict['bounding_box']
            )
            text_obj.confidence = text_dict.get('confidence', 0.0)
            text_obj.properties = text_dict.get('properties', {})
            text_element_objects.append(text_obj)
        
        # 调用原有的分析方法
        return self.analyze_mathematical_semantics(geometry_elements, text_element_objects, [])

if __name__ == "__main__":
    # 测试代码
    print("测试数学语义分析器...")
    
    # 创建测试几何元素
    triangle = Triangle([Point(0, 0), Point(100, 0), Point(50, 80)])
    triangle.id = "triangle1"
    
    line1 = Line(Point(0, 0), Point(100, 0))
    line1.id = "line1"
    
    line2 = Line(Point(50, -20), Point(50, 100))
    line2.id = "line2"
    
    geometry_elements = [triangle, line1, line2]
    
    # 创建测试文本元素
    text1 = TextElement("△ABC", Point(50, 40), (Point(30, 30), Point(70, 50)))
    text1.id = "text1"
    
    text2 = TextElement("AB", Point(50, -10), (Point(40, -15), Point(60, -5)))
    text2.id = "text2"
    
    text3 = TextElement("90°", Point(60, 10), (Point(55, 5), Point(65, 15)))
    text3.id = "text3"
    
    text_elements = [text1, text2, text3]
    
    # 创建测试关系
    from constraint_solver import GeometricRelationship
    relationships = [
        GeometricRelationship("line1", "line2", "perpendicular", {}, 0.9)
    ]
    
    # 测试语义分析
    analyzer = MathSemanticAnalyzer()
    result = analyzer.analyze_mathematical_semantics(geometry_elements, text_elements, relationships)
    
    print(f"场景图节点数: {result['scene_graph'].number_of_nodes()}")
    print(f"场景图边数: {result['scene_graph'].number_of_edges()}")
    
    print(f"\n文字-几何绑定 ({len(result['text_geometry_bindings'])} 个):")
    for binding in result['text_geometry_bindings']:
        print(f"  {binding}")
    
    print(f"\n数学解释 ({len(result['mathematical_interpretations'])} 个):")
    for interpretation in result['mathematical_interpretations']:
        print(f"  {interpretation.concept.concept_type}: {interpretation.mathematical_meaning}")
        print(f"    置信度: {interpretation.confidence:.3f}")
        print(f"    支持证据: {interpretation.supporting_evidence}")
    
    print(f"\n坐标系信息:")
    coords = result['coordinate_system']
    if coords:
        print(f"  原点: {coords['origin']}")
        print(f"  中心: {coords['center']}")
        print(f"  边界: {coords['bounds']['width']:.1f} x {coords['bounds']['height']:.1f}")
        print(f"  比例: {coords['scale']:.3f}")
    
    print(f"\n置信度指标:")
    metrics = result['confidence_metrics']
    print(f"  平均绑定置信度: {metrics['average_binding_confidence']:.3f}")
    print(f"  平均解释置信度: {metrics['average_interpretation_confidence']:.3f}")
    print(f"  绑定总数: {metrics['total_bindings']}")
    print(f"  解释总数: {metrics['total_interpretations']}")