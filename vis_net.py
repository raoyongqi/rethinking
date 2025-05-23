import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
import pandas as pd


# 配置字体
config = {
    "font.family": 'Arial',
    'font.size': 20,
  
}
train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

if 'hwsd_soil_clm_res_pct_clay' in train_df.columns:
    train_df = train_df.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})


# 创建列名重命名字典
rename_dict = {
    'lat': 'Latitude',
    'lon': 'Longitude',
    'awt_soc': 'Area-weighted Soil Organic Carbon',
    'dom_mu': 'Dominant Mapping Unit ID',
    's_sand': 'Soil Sand',
    't_sand': 'Topsoil Sand',
    'hand': 'Height Above Nearest Drainage',
    'srad': 'Solar Radiation',
    'bio_13': 'Precipitation of Wettest Month',
    'bio_15': 'Precipitation Seasonality',
    'bio_18': 'Precipitation of Warmest Quarter',
    'bio_19': 'Precipitation of Coldest Quarter',
    'bio_3': 'Isothermality',
    'bio_6': 'Min Temperature of Coldest Month',
    'bio_8': 'Mean Temperature of Wettest Quarter',
    'wind': 'Wind Speed'}


rename_dict.update({
    'new\\bio_11': 'Mean Temperature of Coldest Quarter',  
    'new\\bio_13': 'Precipitation of Wettest Month',  # bio_13 重复命名示例
    'new\\bio_15': 'Precipitation Seasonality',
    'new\\bio_18': 'Precipitation of Warmest Quarter',
    'new\\bio_3': 'Isothermality',
    'new\\elev': 'Elevation',
    'new/hand': 'Height Above Nearest Drainage',

    'new\\hand': 'Height Above Nearest Drainage',
    'new\\srad': 'Solar Radiation',
    'new\\wind': 'Wind Speed'
})

train_df.rename(columns=rename_dict, inplace=True)

print(len(train_df.columns))

print(len(set(train_df.columns)))
print(train_df.columns)

# input_layer = set(item for item in  train_df.columns if item !='pathogen load' )

# hidden_layer_1 = ['h1_1', 'h1_2', 'h1_3', 'h1_4', 'h1_5', 'h1_6']

# hidden_layer_2 = ['h2_1', 'h2_2', 'h2_3', 'h2_4', 'h2_5']

# output_layer = ['Pathogen Load']

# for input_node in input_layer:
#     for hidden_node in hidden_layer_1:
#         G.add_edge(input_node, hidden_node)

# for node_h1 in hidden_layer_1:
#     for node_h2 in hidden_layer_2:
#         G.add_edge(node_h1, node_h2)

# for node_h2 in hidden_layer_2:
#     for node_output in output_layer:
#         G.add_edge(node_h2, node_output)

# pos = {}

# input_layer_positions = [(0, i - len(input_layer) // 2) for i in range(len(input_layer))]

# hidden_layer_1_positions = [(1, i - len(hidden_layer_1) // 2) for i in range(len(hidden_layer_1))]

# hidden_layer_2_positions = [(2, i - len(hidden_layer_2) // 2) for i in range(len(hidden_layer_2))]

# output_layer_positions = [(3, 0)]


# for i, node in enumerate(input_layer):
#     pos[node] = input_layer_positions[i]

# for i, node in enumerate(hidden_layer_1):
#     pos[node] = hidden_layer_1_positions[i]

# for i, node in enumerate(hidden_layer_2):
#     pos[node] = hidden_layer_2_positions[i]

# for i, node in enumerate(output_layer):
#     pos[node] = output_layer_positions[i]

# plt.figure(figsize=(16, 8))  # 增加宽度

# pos = {
#     # 将所有节点的x坐标向右移动
#     node: (x, y) for node, (x, y) in pos.items()
# }
# nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lavender')

# nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='black')


# plt.text(-0.5, -10, 'Input Layer', ha='center', va='center', )
# plt.text(1, -10, 'Hidden Layer 1', ha='center', va='center', )
# plt.text(2.5, -10, 'Hidden Layer 2', ha='center', va='center', )
# plt.text(3.5, -10, 'Output Layer', ha='center', va='center', )

# for i, node in enumerate(input_layer):
#     plt.text(-0.5, pos[node][1], node, ha='center', va='center', )



# for i, node in enumerate(output_layer):
#     plt.text(3.5, pos[node][1], node, ha='center', va='center', )

# # 显示图形
# # plt.title("Single-Target Neural Network")

# plt.axis('off')  # 关闭坐标轴
# filename= "data/sg.png"
# plt.savefig(filename, dpi=300)  # 保存高分辨率图片


# plt.show()
