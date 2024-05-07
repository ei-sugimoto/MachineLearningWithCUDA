from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data  # type: ignore
else:
    from torch_geometric.data import Data

# ノード
src = [0, 1, 2]  # 送信側
dst = [1, 2, 1]  # 受信側

# エッジ
edge_index = torch.tensor([src, dst], dtype=torch.long)

# ノードの特徴量
x0 = [1, 2]
x1 = [3, 4]
x2 = [5, 6]
x = torch.tensor([x0, x1, x2], dtype=torch.float)

# ラベル
y0 = [1]
y1 = [0]
y2 = [1]
y = torch.tensor([y0, y1, y2], dtype=torch.float)

data = Data(x=x, y=y, edge_index=edge_index)


def check_graph(data):
    """グラフ情報を表示"""
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data["x"])
    print("====== ノードのクラス:y ======")
    print(data["y"])
    print("========= エッジ形状 =========")
    print(data["edge_index"])


check_graph(data)
