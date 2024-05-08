from typing import TYPE_CHECKING

import torch

from class_timer import control_timer

if TYPE_CHECKING:
    from torch_geometric.data import Data  # type: ignore
else:
    from torch_geometric.data import Data


def graph_convolutional_networks():

    src = [0, 1, 2]
    dst = [1, 2, 1]

    edge_index = torch.tensor([src, dst], dtype=torch.half)

    x0 = [1, 2]
    x1 = [3, 4]
    x2 = [5, 6]
    x = torch.tensor([x0, x1, x2], dtype=torch.half)

    y0 = [1]
    y1 = [0]
    y2 = [1]
    y = torch.tensor([y0, y1, y2], dtype=torch.half)

    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def check_graph(data):
    """グラフ情報を表示"""
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.has_self_loops())
    print("自己ループの有無:", data.has_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data["x"])
    print("====== ノードのクラス:y ======")
    print(data["y"])
    print("========= エッジ形状 =========")
    print(data["edge_index"])


timer = control_timer()

timer.start()
data = graph_convolutional_networks()
timer.end()
print("実行時間:", timer.get_time() * 10**3, "ms")
check_graph(data)
