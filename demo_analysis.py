"""
PDB复合物分析 - 快速演示脚本
============================
这个脚本展示如何快速分析您的PDB文件

运行方式:
    python demo_analysis.py
"""

from pdb_complex_analyzer import PDBComplexAnalyzer, BatchPDBAnalyzer
from pathlib import Path
import sys


def demo_single_file_analysis():
    """演示1: 单个文件分析"""
    print("\n" + "="*80)
    print("演示1: 单个PDB文件详细分析")
    print("="*80)
    
    # 指定一个PDB文件
    pdb_dir = Path("complex-20251129T063258Z-1-001/complex")
    
    # 获取第一个PDB文件
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print("错误: 找不到PDB文件!")
        print(f"请确保目录存在: {pdb_dir}")
        return
    
    sample_file = str(pdb_files[0])
    print(f"\n分析文件: {Path(sample_file).name}")
    
    # 创建分析器
    analyzer = PDBComplexAnalyzer(sample_file)
    analyzer.parse()
    
    # 获取基本统计
    stats = analyzer.get_statistics()
    
    print("\n【基本信息】")
    print(f"  文件名: {stats['filename']}")
    print(f"  蛋白质原子数: {stats['n_protein_atoms']}")
    print(f"  配体原子数: {stats['n_ligand_atoms']}")
    print(f"  序列长度: {stats['sequence_length']} 个残基")
    
    print("\n【配体组成】")
    for element, count in sorted(stats['ligand_composition'].items()):
        print(f"  元素 {element}: {count} 个原子")
    
    print(f"\n【空间信息】")
    print(f"  蛋白质质心: ({stats['protein_com'][0]:.2f}, "
          f"{stats['protein_com'][1]:.2f}, {stats['protein_com'][2]:.2f}) Å")
    print(f"  配体质心: ({stats['ligand_com'][0]:.2f}, "
          f"{stats['ligand_com'][1]:.2f}, {stats['ligand_com'][2]:.2f}) Å")
    print(f"  质心间距离: {stats['com_distance']:.2f} Å")
    
    # 查找相互作用
    print("\n【相互作用分析】")
    interactions = analyzer.find_interactions(distance_cutoff=4.0)
    print(f"  发现 {len(interactions)} 个蛋白-配体相互作用 (距离 < 4.0 Å)")
    
    if interactions:
        print("\n  最近的5个相互作用:")
        print(f"  {'残基':<15} {'蛋白原子':<12} {'配体原子':<12} {'距离(Å)':<10}")
        print("  " + "-"*50)
        for p_atom, l_atom, dist in interactions[:5]:
            print(f"  {p_atom.resname}{p_atom.resseq:<11} "
                  f"{p_atom.name:<12} {l_atom.name:<12} {dist:<10.3f}")
    
    # 结合位点分析
    binding_residues = analyzer.get_binding_residues(distance_cutoff=4.0)
    print(f"\n【结合位点】")
    print(f"  参与结合的残基数: {len(binding_residues)}")
    
    if binding_residues:
        sorted_residues = sorted(
            binding_residues.items(),
            key=lambda x: x[1]['min_distance']
        )
        
        print("\n  前10个关键结合残基:")
        print(f"  {'残基':<12} {'链':<6} {'最近距离(Å)':<15} {'接触数':<10}")
        print("  " + "-"*50)
        for resseq, info in sorted_residues[:10]:
            print(f"  {info['resname']} {resseq:<8} {info['chain']:<6} "
                  f"{info['min_distance']:<15.3f} {len(info['contacts']):<10}")
    
    # 生成详细报告
    report_file = f"report_{Path(sample_file).stem}.txt"
    analyzer.generate_report(report_file)
    print(f"\n  ✓ 详细报告已保存至: {report_file}")


def demo_batch_analysis():
    """演示2: 批量文件分析"""
    print("\n" + "="*80)
    print("演示2: 批量PDB文件分析")
    print("="*80)
    
    pdb_dir = Path("complex-20251129T063258Z-1-001/complex")
    
    if not pdb_dir.exists():
        print(f"错误: 目录不存在: {pdb_dir}")
        return
    
    print(f"\n正在分析目录: {pdb_dir}")
    
    # 创建批量分析器
    batch_analyzer = BatchPDBAnalyzer(str(pdb_dir))
    
    # 只加载前10个文件作为演示（完整分析可以去掉限制）
    pdb_files = sorted(pdb_dir.glob("*.pdb"))[:10]
    print(f"加载前10个PDB文件进行演示...")
    
    for pdb_file in pdb_files:
        print(f"  加载: {pdb_file.name}")
        analyzer = PDBComplexAnalyzer(str(pdb_file))
        analyzer.parse()
        batch_analyzer.analyzers.append(analyzer)
    
    print(f"\n成功加载 {len(batch_analyzer.analyzers)} 个文件\n")
    
    # 生成汇总表
    print("【统计汇总表】")
    print(f"{'文件名':<25} {'蛋白原子':<12} {'配体原子':<12} {'结合残基':<12} {'质心距离(Å)':<15}")
    print("-" * 85)
    
    results = []
    for analyzer in batch_analyzer.analyzers:
        stats = analyzer.get_statistics()
        binding_res = analyzer.get_binding_residues()
        
        print(f"{stats['filename']:<25} "
              f"{stats['n_protein_atoms']:<12} "
              f"{stats['n_ligand_atoms']:<12} "
              f"{len(binding_res):<12} "
              f"{stats['com_distance']:<15.2f}")
        
        results.append({
            'filename': stats['filename'],
            'n_binding_res': len(binding_res),
            'com_distance': stats['com_distance']
        })
    
    # 统计分析
    n_binding_residues = [r['n_binding_res'] for r in results]
    com_distances = [r['com_distance'] for r in results]
    
    print(f"\n【统计摘要】")
    print(f"  结合残基数量:")
    print(f"    平均: {sum(n_binding_residues)/len(n_binding_residues):.1f}")
    print(f"    最小: {min(n_binding_residues)}")
    print(f"    最大: {max(n_binding_residues)}")
    
    print(f"  质心距离:")
    print(f"    平均: {sum(com_distances)/len(com_distances):.2f} Å")
    print(f"    最小: {min(com_distances):.2f} Å")
    print(f"    最大: {max(com_distances):.2f} Å")
    
    # 找出最佳候选
    print(f"\n【最佳候选】")
    results.sort(key=lambda x: x['n_binding_res'], reverse=True)
    
    print("  结合残基数量最多的3个复合物:")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['filename']:<25} - {r['n_binding_res']} 个结合残基")
    
    # 保存汇总
    summary_file = "batch_summary.txt"
    batch_analyzer.generate_summary_table(summary_file)
    print(f"\n  ✓ 完整汇总表已保存至: {summary_file}")


def demo_interaction_details():
    """演示3: 详细相互作用分析"""
    print("\n" + "="*80)
    print("演示3: 详细相互作用类型分析")
    print("="*80)
    
    pdb_dir = Path("complex-20251129T063258Z-1-001/complex")
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    
    if not pdb_files:
        print("错误: 找不到PDB文件!")
        return
    
    sample_file = str(pdb_files[0])
    print(f"\n分析文件: {Path(sample_file).name}")
    
    analyzer = PDBComplexAnalyzer(sample_file)
    analyzer.parse()
    
    # 分析不同距离范围的相互作用
    print("\n【距离分布分析】")
    
    distance_ranges = [
        (0, 2.5, "很近 (可能氢键)"),
        (2.5, 3.5, "氢键范围"),
        (3.5, 4.5, "范德华接触"),
        (4.5, 5.5, "弱相互作用")
    ]
    
    interactions = analyzer.find_interactions(distance_cutoff=5.5)
    
    for min_dist, max_dist, description in distance_ranges:
        count = sum(1 for _, _, d in interactions if min_dist <= d < max_dist)
        if count > 0:
            print(f"  {description} ({min_dist}-{max_dist} Å): {count} 个")
    
    # 氨基酸类型统计
    print("\n【结合位点氨基酸分类】")
    
    binding_residues = analyzer.get_binding_residues()
    
    aa_categories = {
        '极性': {'SER', 'THR', 'ASN', 'GLN', 'TYR'},
        '带正电': {'LYS', 'ARG', 'HIS'},
        '带负电': {'ASP', 'GLU'},
        '疏水': {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'},
        '特殊': {'GLY', 'CYS'}
    }
    
    category_counts = {cat: 0 for cat in aa_categories}
    
    for resseq, info in binding_residues.items():
        resname = info['resname']
        for category, residues in aa_categories.items():
            if resname in residues:
                category_counts[category] += 1
                break
    
    for category, count in category_counts.items():
        if count > 0:
            print(f"  {category}: {count} 个残基")


def demo_ligand_analysis():
    """演示4: 配体详细分析"""
    print("\n" + "="*80)
    print("演示4: 配体分子详细分析")
    print("="*80)
    
    pdb_dir = Path("complex-20251129T063258Z-1-001/complex")
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    
    if not pdb_files:
        print("错误: 找不到PDB文件!")
        return
    
    sample_file = str(pdb_files[0])
    print(f"\n分析文件: {Path(sample_file).name}")
    
    analyzer = PDBComplexAnalyzer(sample_file)
    analyzer.parse()
    
    print("\n【配体原子详情】")
    print(f"  总原子数: {len(analyzer.ligand_atoms)}")
    
    # 元素统计
    composition = analyzer.analyze_ligand_composition()
    print("\n  元素组成:")
    for element, count in sorted(composition.items()):
        print(f"    {element:>2}: {count:>3} 个原子")
    
    # 化学键统计
    print(f"\n【化学键信息】")
    print(f"  总键数: {len(analyzer.bonds)}")
    
    bond_orders = {}
    for bond in analyzer.bonds:
        bond_orders[bond.bond_order] = bond_orders.get(bond.bond_order, 0) + 1
    
    bond_names = {1: '单键', 2: '双键', 3: '三键'}
    for order, count in sorted(bond_orders.items()):
        print(f"  {bond_names.get(order, f'{order}键')}: {count} 个")
    
    # 配体空间范围
    ligand_atoms = analyzer.ligand_atoms
    x_coords = [atom.x for atom in ligand_atoms]
    y_coords = [atom.y for atom in ligand_atoms]
    z_coords = [atom.z for atom in ligand_atoms]
    
    print("\n【配体空间范围】")
    print(f"  X轴: {min(x_coords):.2f} 到 {max(x_coords):.2f} Å "
          f"(跨度: {max(x_coords)-min(x_coords):.2f} Å)")
    print(f"  Y轴: {min(y_coords):.2f} 到 {max(y_coords):.2f} Å "
          f"(跨度: {max(y_coords)-min(y_coords):.2f} Å)")
    print(f"  Z轴: {min(z_coords):.2f} 到 {max(z_coords):.2f} Å "
          f"(跨度: {max(z_coords)-min(z_coords):.2f} Å)")
    
    # 可能的分子类型推断
    print("\n【分子类型推断】")
    if 'P' in composition and 'N' in composition:
        if composition.get('P', 0) >= 2:
            print("  可能是核苷酸类分子 (如 ATP, ADP, GTP等)")
        else:
            print("  可能是核苷酸或磷酸化合物")
    elif 'N' in composition and 'O' in composition:
        print("  可能是有机小分子 (含氮氧)")
    else:
        print("  需要进一步分析确定分子类型")


def main():
    """主函数 - 运行所有演示"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PDB复合物分析工具 - 演示程序" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # 运行所有演示
        demo_single_file_analysis()
        demo_batch_analysis()
        demo_interaction_details()
        demo_ligand_analysis()
        
        print("\n" + "="*80)
        print("所有演示完成!")
        print("="*80)
        
        print("\n【生成的文件】")
        print("  - report_*.txt: 单个文件的详细分析报告")
        print("  - batch_summary.txt: 批量分析汇总表")
        
        print("\n【下一步建议】")
        print("  1. 查看生成的报告文件")
        print("  2. 运行完整批量分析: python pdb_complex_analyzer.py complex-20251129T063258Z-1-001/complex/ -b")
        print("  3. 分析特定文件: python pdb_complex_analyzer.py <文件路径>")
        print("  4. 阅读详细文档: PDB复合物分析指南.md")
        
        print("\n" + "="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请确保以下条件满足:")
        print("  1. 当前目录下存在 'complex-20251129T063258Z-1-001/complex/' 文件夹")
        print("  2. 该文件夹中包含PDB文件")
        print("  3. pdb_complex_analyzer.py 文件存在")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
