import matplotlib.pyplot as plt
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 输入层（每列居中对齐）
input_layer = ['lat', 'lon', 'awt_soc', 'dom_mu', 'pct_clay', 's_sand', 'elev', 'hand', 'srad', 'wind', 'bio_11', 'bio_13', 'bio_15', 'bio_18', 'bio_3', 'tmax']

# 隐藏层1
hidden_layer_1 = ['h1_1', 'h1_2', 'h1_3', 'h1_4', 'h1_5', 'h1_6']

# 隐藏层2
hidden_layer_2 = ['h2_1', 'h2_2', 'h2_3', 'h2_4', 'h2_5']

# 输出层
output_layer = ['pathogen load']

# 添加输入层到隐藏层1的连接
for input_node in input_layer:
    for hidden_node in hidden_layer_1:
        G.add_edge(input_node, hidden_node)

# 添加隐藏层1到隐藏层2的连接
for node_h1 in hidden_layer_1:
    for node_h2 in hidden_layer_2:
        G.add_edge(node_h1, node_h2)

# 添加隐藏层2到输出层的连接
for node_h2 in hidden_layer_2:
    for node_output in output_layer:
        G.add_edge(node_h2, node_output)

# 设置节点位置
pos = {}

# 输入层的位置
input_layer_positions = [(0, i - len(input_layer) // 2) for i in range(len(input_layer))]

# 隐藏层1的位置
hidden_layer_1_positions = [(1, i - len(hidden_layer_1) // 2) for i in range(len(hidden_layer_1))]

# 隐藏层2的位置
hidden_layer_2_positions = [(2, i - len(hidden_layer_2) // 2) for i in range(len(hidden_layer_2))]

# 输出层的位置
output_layer_positions = [(3, 0)]  # 输出层通常是一个节点，放在中间

# 合并所有层的位置
for i, node in enumerate(input_layer):
    pos[node] = input_layer_positions[i]

for i, node in enumerate(hidden_layer_1):
    pos[node] = hidden_layer_1_positions[i]

for i, node in enumerate(hidden_layer_2):
    pos[node] = hidden_layer_2_positions[i]

for i, node in enumerate(output_layer):
    pos[node] = output_layer_positions[i]

# 绘制网络
plt.figure(figsize=(12, 8))

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lavender')

# 绘制边
nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='black')

# 绘制节点标签
# nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# 添加层标签（输入层、隐藏层1、隐藏层2、输出层）
plt.text(-0.5, -10, 'Input Layer', fontsize=12, ha='center', va='center', fontweight='bold')
plt.text(1, -10, 'Hidden Layer 1', fontsize=12, ha='center', va='center', fontweight='bold')
plt.text(2.5, -10, 'Hidden Layer 2', fontsize=12, ha='center', va='center', fontweight='bold')
plt.text(3.5, -10, 'Output Layer', fontsize=12, ha='center', va='center', fontweight='bold')
# 调整每层的标签位置，向前偏移
for i, node in enumerate(input_layer):
    plt.text(-0.5, pos[node][1], node, fontsize=12, ha='center', va='center', fontweight='bold')



for i, node in enumerate(output_layer):
    plt.text(3.5, pos[node][1], node, fontsize=12, ha='center', va='center', fontweight='bold')

# 显示图形
plt.title("Single-Target Neural Network", fontsize=16)

plt.axis('off')  # 关闭坐标轴
filename= "data/sg.png"
plt.savefig(filename, dpi=300)  # 保存高分辨率图片


plt.show()
