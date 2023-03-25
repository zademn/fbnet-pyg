import torch
from torch import nn
from torch_cluster import knn_graph
from einops import rearrange, repeat
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import EdgeConv, DynamicEdgeConv


def edge_conv_nn(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(2 * in_channels, out_channels // 2),
        nn.BatchNorm1d(out_channels // 2),
        nn.LeakyReLU(0.2),
        nn.Linear(out_channels // 2, out_channels),
    )

class PointShuffle(torch.nn.Module):
    def __init__(self, r):
        """
        Shuffles [N, r * C] -> [r * N, C]
        Args:
            r: int
                scale
        """
        super().__init__()
        # Config
        self.r = r

    def forward(self, x, batch=None):
        """
        Args:
            x: Tensor
                Node feature matrix of all point clouds concatenated [N, r * C]
            batch: Optional[LongTensor]
                batch tensor [N, ] as described in PyG docs.
                For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        r = self.r

        n, c = x.shape[0], x.shape[1]

        if batch is not None:
            # split into batches.
            x, _ = to_dense_batch(
                x, batch=batch
            )  # [N, C] -> [B, N_, C] where N = B * N_
            # Split the channels dim c = (c2, r). Then combine (n r).
            x = rearrange(x, "b n (c2 r) -> b (n r) c2", c2=c // r)
            # combine back
            x = rearrange(x, "b n c -> (b n) c")

        else:
            x = rearrange(x, "n (c2 r) -> (n r) c2", c2=c // r)

        return x


class NodeShuffle(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        r: int,
        conv="edge",
    ):
        """
        Transforms input: [N, C] -> [r * N, C']
        Parameters:
        ----------
        in_channels: int
            number of input channels C
        out_channels: int
            number of output channels C'
        k: int
            number of neighbours to sample
        r: int
            upsampling ratio
        """
        super(NodeShuffle, self).__init__()

        # Config
        self.k = k
        self.r = r
        self.in_channels = in_channels
        # Layers
        # self.gcn = GraphConv(
        #     in_channels=in_channels, out_channels=in_channels * r, conv=conv
        # )  # self.gcn = DynConv(
        #     kernel_size=k,
        #     dilation=1,
        #     in_channels=in_channels,
        #     out_channels=in_channels * r,
        #     conv="edge",
        #     knn="matrix",
        # )
        self.gcn = DynamicEdgeConv(nn=edge_conv_nn(in_channels, in_channels * r), k=k)
        self.ps = PointShuffle(r=r)
        self.lin = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(
        self, x, edge_index=None, pos=None, batch=None, return_batch: bool = False
    ):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]
        edge_index: Tensor, default = None
            Edge index of shape [N * k, 2]. If it's not provided the graph is computed dynamically.
        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        return_batch: bool, default, = False
            True - will return the upsampled batch vector.
        """

        # if edge_index is None:
        #     edge_index = knn_graph(x, self.k, batch=batch)
        x = self.gcn(x)  # [N, C] -> [N, r * C]
        # x = self.gcn(x, batch=batch)
        x = self.ps(x, batch=batch)  # [N, r * C] -> [r * N, C]
        x = self.lin(x)  # [r * N, C] -> [r * N, C']

        if return_batch:
            if batch is not None:
                batch_ = batch.repeat_interleave(self.r)  # repeat each number r times
            else:
                batch_ = torch.zeros(len(x))
            return x, batch_
        return x