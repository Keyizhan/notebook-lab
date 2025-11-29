# PDB蛋白质-配体复合物分析工具包

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy matplotlib
```

### 2. 运行演示

```bash
# 运行完整演示程序（推荐新手）
python demo_analysis.py

# 单个文件分析
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/104.pdb

# 批量分析所有文件
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/ -b -o summary.txt
```

### 3. 查看结果

运行后会生成：
- `report_*.txt` - 详细分析报告
- `summary.txt` - 批量分析汇总表
- `statistics.png` - 统计图表（批量模式）

---

## 📁 文件说明

### 核心文件

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `pdb_complex_analyzer.py` | 主分析程序 | 提供所有分析功能的核心代码 |
| `demo_analysis.py` | 演示脚本 | 快速上手示例，展示常见用法 |
| `PDB复合物分析指南.md` | 详细文档 | 完整的使用教程和PDB格式讲解 |
| `README_PDB分析.md` | 本文件 | 快速参考指南 |

### 数据文件

```
complex-20251129T063258Z-1-001/
└── complex/
    ├── 104.pdb
    ├── 115.pdb
    ├── 1033.pdb
    └── ... (共3432个PDB文件)
```

---

## 💡 常用命令

### 命令行模式

```bash
# 1. 分析单个文件并保存报告
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/104.pdb -o report.txt

# 2. 设置相互作用距离阈值为3.5埃
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/104.pdb -d 3.5

# 3. 批量分析并生成图表
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/ -b

# 4. 批量分析并指定输出文件
python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/ -b -o my_summary.txt
```

### Python脚本模式

```python
from pdb_complex_analyzer import PDBComplexAnalyzer

# 单文件分析
analyzer = PDBComplexAnalyzer("path/to/file.pdb")
analyzer.parse()

# 获取统计信息
stats = analyzer.get_statistics()
print(f"蛋白原子数: {stats['n_protein_atoms']}")
print(f"配体原子数: {stats['n_ligand_atoms']}")

# 查找相互作用
interactions = analyzer.find_interactions(distance_cutoff=4.0)
print(f"发现 {len(interactions)} 个相互作用")

# 获取结合残基
binding_res = analyzer.get_binding_residues()
print(f"结合残基数: {len(binding_res)}")

# 生成报告
analyzer.generate_report("report.txt")
```

```python
from pdb_complex_analyzer import BatchPDBAnalyzer

# 批量分析
batch = BatchPDBAnalyzer("complex-20251129T063258Z-1-001/complex/")
batch.load_all_pdbs()

# 生成汇总
batch.generate_summary_table("summary.txt")

# 绘制图表
batch.plot_statistics("stats.png")

# 比较结合位点
comparison = batch.compare_binding_sites()
print(f"共同结合残基: {comparison['common_residues']}")
```

---

## 📊 主要功能

### 1. 基本分析

- ✅ 解析PDB文件结构
- ✅ 提取蛋白质和配体原子坐标
- ✅ 统计原子数量和组成
- ✅ 计算质心位置

### 2. 相互作用分析

- ✅ 识别蛋白-配体相互作用
- ✅ 计算原子间距离
- ✅ 鉴定结合位点残基
- ✅ 统计接触数量

### 3. 批量处理

- ✅ 同时处理3432个PDB文件
- ✅ 生成汇总统计表
- ✅ 比较不同复合物
- ✅ 识别保守结合位点

### 4. 可视化

- ✅ 生成统计图表
- ✅ 质心距离分布
- ✅ 结合残基数量分布

---

## 🔍 核心概念

### PDB文件包含什么？

您的PDB文件包含：
1. **蛋白质结构** - 约250个氨基酸，约3979个原子
2. **配体分子** - 约52个原子，可能是ATP或类似核苷酸
3. **相互作用信息** - 蛋白质如何与小分子结合

### 关键指标

| 指标 | 含义 | 理想范围 |
|------|------|---------|
| 结合残基数 | 参与结合的氨基酸数量 | 10-30个 |
| 质心距离 | 蛋白与配体中心距离 | 10-20 Å |
| 相互作用数 | 原子间接触数量 | 50-150个 |
| 氢键数 | 潜在氢键数量 | 2-8个 |

---

## 📖 输出解读

### 单文件报告示例

```
================================================================================
PDB复合物分析报告: 104.pdb
================================================================================

【基本信息】
  文件名: 104.pdb
  蛋白质原子数: 3979
  配体原子数: 52
  序列长度: 250 残基

【配体信息】
  元素组成: {'O': 18, 'P': 2, 'C': 20, 'N': 10}
  配体质心坐标: (4.120, 11.340, -0.560)

【结合位点分析】
  结合残基数量: 25
  结合残基列表（按距离排序）:
    SER  45 (链 A) - 最近距离: 2.65 Å, 接触数: 3
    LYS  89 (链 A) - 最近距离: 2.79 Å, 接触数: 2
    THR 123 (链 A) - 最近距离: 2.89 Å, 接触数: 4
    ...
```

### 批量分析汇总示例

```
文件名                      蛋白原子      配体原子      结合残基      质心距离
----------------------------------------------------------------------------------
104.pdb                    3979         52           25           18.45
115.pdb                    3979         52           28           17.23
1033.pdb                   3979         52           22           19.67
...
```

---

## 🎯 常见应用场景

### 场景1: 找出最佳对接结果

```python
from pdb_complex_analyzer import BatchPDBAnalyzer

batch = BatchPDBAnalyzer("complex-20251129T063258Z-1-001/complex/")
batch.load_all_pdbs()

# 收集数据
results = []
for analyzer in batch.analyzers:
    stats = analyzer.get_statistics()
    binding_res = analyzer.get_binding_residues()
    
    results.append({
        'file': stats['filename'],
        'score': len(binding_res),  # 用结合残基数作为评分
        'distance': stats['com_distance']
    })

# 排序
results.sort(key=lambda x: x['score'], reverse=True)

# 显示前10
print("Top 10最佳对接结果:")
for i, r in enumerate(results[:10], 1):
    print(f"{i}. {r['file']}: 结合残基={r['score']}, 距离={r['distance']:.2f}Å")
```

### 场景2: 识别保守结合位点

```python
batch = BatchPDBAnalyzer("complex-20251129T063258Z-1-001/complex/")
batch.load_all_pdbs()

comparison = batch.compare_binding_sites()
common = comparison['common_residues']

print(f"所有复合物共同的结合残基: {sorted(common)}")
print(f"这些残基可能对结合至关重要!")
```

### 场景3: 分析特定残基的作用

```python
analyzer = PDBComplexAnalyzer("complex-20251129T063258Z-1-001/complex/104.pdb")
analyzer.parse()

# 查找残基45的所有相互作用
interactions = analyzer.find_interactions()
res45_interactions = [
    (p, l, d) for p, l, d in interactions if p.resseq == 45
]

print(f"残基45的相互作用:")
for p_atom, l_atom, dist in res45_interactions:
    print(f"  {p_atom.name} -- {l_atom.name}: {dist:.2f} Å")
```

---

## ⚠️ 注意事项

1. **文件路径**: 确保PDB文件路径正确
2. **内存使用**: 批量分析3432个文件需要较多内存（约1-2GB）
3. **处理时间**: 完整批量分析可能需要5-10分钟
4. **距离阈值**: 默认4.0Å，可根据需要调整

---

## 🆘 常见问题

### Q: 为什么有3432个PDB文件？
**A**: 这是分子对接虚拟筛选的结果，每个文件代表一个蛋白-配体复合物结构。

### Q: 如何判断对接质量？
**A**: 主要看：
- 结合残基数量（10-30个较好）
- 相互作用类型（有氢键更好）
- 质心距离（10-20Å合理）
- 原子间无严重冲突

### Q: 配体是什么分子？
**A**: 根据元素组成（含P、N、O），很可能是ATP、ADP或类似核苷酸。

### Q: 如何可视化结构？
**A**: 推荐使用：
- PyMOL: `conda install -c conda-forge pymol-open-source`
- ChimeraX: https://www.rbvi.ucsf.edu/chimerax/
- VMD: https://www.ks.uiuc.edu/Research/vmd/

### Q: 程序运行太慢怎么办？
**A**: 
- 先分析少量文件测试
- 使用多进程加速（需修改代码）
- 只分析关键指标

---

## 📚 进阶学习

### 推荐阅读顺序

1. ✅ **本文件 (README)** - 快速上手
2. ✅ **demo_analysis.py** - 运行示例代码
3. ✅ **PDB复合物分析指南.md** - 深入理解
4. ✅ **pdb_complex_analyzer.py** - 源代码学习

### 相关资源

- PDB格式官方文档: https://www.wwpdb.org/documentation/file-format
- 分子对接教程: http://autodock.scripps.edu/
- Python结构生物学: https://biopython.org/

---

## 🤝 技术支持

如有问题，请检查：
1. Python版本 >= 3.7
2. 依赖包已安装: `pip list | grep -E "numpy|matplotlib"`
3. 文件路径正确
4. PDB文件完整无损

---

## 📝 更新日志

### v1.0 (2024-11-29)
- ✨ 初始版本
- ✨ 支持PDB文件解析
- ✨ 相互作用分析
- ✨ 批量处理功能
- ✨ 统计图表生成

---

**祝您分析愉快！🎉**

如需更多帮助，请查看 `PDB复合物分析指南.md` 获取详细文档。
