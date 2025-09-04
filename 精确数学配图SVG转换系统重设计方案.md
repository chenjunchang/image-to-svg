精确数学配图SVG转换系统重设计方案

     问题诊断

     通过对比原图和当前输出，发现关键问题：
     1. 过度碎片化：生成105条短线段而非几何图元（三角形、直线）
     2. 缺乏几何理解：无法识别基本形状和几何关系
     3. 文字识别失败：OCR完全失效（0个文字）
     4. 缺乏约束推理：无法理解点、线、面的几何约束关系

     核心方案：几何语义理解架构

     架构设计（4层系统）

     第1层：高级几何图元检测 GeometryPrimitiveDetector

     基于OpenCV高级算法，替代低级像素处理
     class GeometryPrimitiveDetector:
         - detect_polygons(): 使用contour+approxPolyDP检测三角形/四边形
         - detect_lines(): Hough直线检测+智能合并
         - detect_circles(): HoughCircles精确圆形检测
         - detect_points(): 角点检测+几何验证
         - classify_shapes(): 基于顶点数和几何特征分类

     第2层：几何约束推理引擎 ConstraintSolver

     参考CAD系统的几何约束求解
     class ConstraintSolver:
         - analyze_relationships(): 识别平行/垂直/相交关系
         - build_constraint_graph(): 构建几何约束图
         - solve_positions(): 约束求解优化坐标
         - validate_geometry(): 几何一致性检查

     第3层：数学语义理解 MathSemanticAnalyzer

     专门针对数学图表的语义解析
     class MathSemanticAnalyzer:
         - build_scene_graph(): 构建几何场景图
         - integrate_text_geometry(): 文字-几何元素绑定
         - infer_mathematical_meaning(): 推断数学含义
         - establish_coordinate_system(): 建立数学坐标系

     第4层：精确SVG重建器 PreciseSVGReconstructor

     生成数学级精度的结构化SVG
     class PreciseSVGReconstructor:
         - generate_geometric_svg(): 使用真正几何图元
         - optimize_coordinates(): 坐标精度优化
         - structure_output(): 层次化SVG结构
         - validate_mathematical_accuracy(): 数学精度验证

     关键技术改进

     1. 智能几何检测

     - 多边形识别：contour + Douglas-Peucker算法精确识别三角形
     - 直线优化：Hough检测 + 线段合并 + 延伸算法
     - 点位精确定位：Harris角点 + 几何约束优化

     2. 高级OCR集成

     - 区域分割：几何掩码引导的OCR区域提取
     - 数学字符优化：专门的数学符号识别
     - 位置绑定：文字与几何元素的精确关联

     3. 几何约束推理

     - 关系识别：自动识别平行、垂直、相交关系
     - 约束求解：使用数值优化方法精确定位
     - 一致性检查：确保几何关系的数学正确性

     4. 数学级SVG输出

     - 真正几何图元：<polygon>, <line>, <circle>替代复杂路径
     - 精确坐标系：保持数学比例和精度
     - 结构化输出：分层组织（几何层、文字层、标注层）

     实现文件结构

     precise_math_svg_converter.py     # 主转换器
     ├─ geometry_detector.py           # 几何图元检测
     ├─ constraint_solver.py           # 约束推理引擎
     ├─ math_semantic_analyzer.py      # 数学语义分析
     ├─ precise_svg_generator.py       # 精确SVG生成
     └─ math_config.py                # 数学图表配置