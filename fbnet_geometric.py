import torch
import torch_geometric
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch_geometric.nn import (
    MessagePassing,
    knn,
    fps,
    DynamicEdgeConv,
    global_max_pool,
    global_mean_pool,
    MLP,
)
from torch_geometric.utils import to_dense_batch
import upsample


def get_batch(pcd):
    assert pcd.ndim == 3
    b, n, _ = pcd.shape
    batch = torch.repeat_interleave(torch.arange(b), repeats=n).type(torch.long)
    return batch.to(pcd.device)


def fps_subsample(pcd, n_points: int, random_start: bool = False):
    b, n, _ = pcd.shape
    batch = get_batch(pcd)
    pcd = rearrange(pcd, "b n c -> (b n) c")
    idxs = fps(pcd, batch=batch, ratio=n_points / n, random_start=False)
    return to_dense_batch(pcd[idxs], batch=batch[idxs])[0]


class CrossTransformer(MessagePassing):
    def __init__(
        self,
        in_channel: int,
        pos_hidden_dim: int,
        attn_hidden_dim: int,
        k: int = 2,
    ):
        super().__init__(aggr="add", flow="target_to_source")
        self.k = k
        self.pos_mlp = nn.Sequential(
            nn.Linear(in_features=3, out_features=pos_hidden_dim),
            nn.BatchNorm1d(pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=pos_hidden_dim, out_features=in_channel),
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(in_channel, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, in_channel),
        )

    def forward(self, px, py, fx, fy, edge_index=None, batch_x=None, batch_y=None):

        # Include self in target point cloud
        p_fusion = torch.cat([px, py])
        f_fusion = torch.cat([fx, fy])
        if batch_x is not None:
            fusion_batch = torch.cat([batch_x, batch_y])
        else:
            fusion_batch = None

        if edge_index is None:
            edge_index = knn(
                x=p_fusion,
                y=px,
                batch_x=fusion_batch,
                batch_y=batch_x,
                k=self.k,
            )

        # flow = "target_to_source" => (x_i, x_j), (pos_i, pos_j)
        out = self.propagate(edge_index, x=(fx, f_fusion), pos=(px, p_fusion))
        return out

    def message(self, x_i, x_j, pos_i, pos_j, index):
        # Positional embedding
        delta_ij = self.pos_mlp(pos_i - pos_j)
        # Attention embedding
        attn_weights = self.attn_mlp(x_i - x_j + delta_ij)
        # Normalize attention
        attn_weights = torch_geometric.utils.softmax(
            attn_weights, index=index, num_nodes=None
        )

        # Multiply with the attention weights
        out = attn_weights * (x_j + delta_ij)
        return out


class AdaptGraphPooling(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        pooling_rate: int = 0.25,
        pos_hidden_dim: int = 16,
        attn_hidden_dim: int = 16,
        k=16,
    ):
        super().__init__(aggr="add", flow="target_to_source")
        self.pooling_rate = pooling_rate
        self.k = k
        self.pos_mlp = nn.Sequential(
            nn.Linear(in_features=3, out_features=pos_hidden_dim),
            nn.BatchNorm1d(pos_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=pos_hidden_dim, out_features=in_channels),
        )

        self.feat_attn_mlp = nn.Sequential(
            nn.Linear(in_channels, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(attn_hidden_dim, in_channels),
        )

        self.pos_attn_mlp = nn.Sequential(
            nn.Linear(in_channels, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(attn_hidden_dim, 3),
        )

    def forward(self, x, pos, batch=None, return_batch=True):

        # n_points = len(pos)  # careful with batching
        idxs = fps(pos, batch=batch, ratio=self.pooling_rate, random_start=False)
        pos_ = pos[idxs]
        x_ = x[idxs]

        if batch is not None:
            batch_ = batch[idxs]
        else:
            batch_ = None

        edge_index = knn(pos, pos_, batch_x=batch, batch_y=batch_, k=self.k)
        out = self.propagate(edge_index, x=(x_, x), pos=(pos_, pos))

        new_pos, new_feat = out[:, :3], out[:, 3:]

        if return_batch:
            return new_pos, new_feat, batch_
        else:
            return new_pos, new_feat

    def message(self, x_i, x_j, pos_i, pos_j, index):
        # Positional embedding
        delta_ij = self.pos_mlp(pos_i - pos_j)

        # Positional weights
        pos_weights = self.pos_attn_mlp(x_i - x_j + delta_ij)
        pos_weights = torch_geometric.utils.softmax(
            pos_weights, index=index, num_nodes=None
        )

        # Feature weights
        feat_weights = self.feat_attn_mlp(x_i - x_j + delta_ij)
        feat_weights = torch_geometric.utils.softmax(
            feat_weights, index=index, num_nodes=None
        )

        # Concatenate to return
        out = torch.cat([pos_weights * pos_j, feat_weights * (x_j + delta_ij)], dim=-1)

        return out


def edge_conv_nn(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(2 * in_channels, out_channels // 2),
        nn.BatchNorm1d(out_channels // 2),
        nn.LeakyReLU(0.2),
        nn.Linear(out_channels // 2, out_channels),
    )


class HGNet(nn.Module):
    def __init__(self, num_pc: int = 128, k=3):
        super().__init__()

        # HGNet econder
        self.num_pc = num_pc

        self.gcn_1 = DynamicEdgeConv(nn=edge_conv_nn(3, 64), k=k)
        self.graph_pooling_1 = AdaptGraphPooling(in_channels=64, pooling_rate=0.25, k=k)
        self.gcn_2 = DynamicEdgeConv(nn=edge_conv_nn(64, 128), k=k)
        self.graph_pooling_2 = AdaptGraphPooling(in_channels=128, pooling_rate=0.5, k=k)
        self.gcn_3 = DynamicEdgeConv(nn=edge_conv_nn(128, 256), k=k)

        # Fully-connected decoder
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 2, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=3 * num_pc),
        )

    def forward(self, pos, batch=None, return_batch=True):
        device = pos.device
        x1 = self.gcn_1(pos, batch=batch)

        vertices_pool_1, x1, batch = self.graph_pooling_1(x1, pos, batch=batch)

        x2 = self.gcn_2(x1, batch=batch)

        vertices_pool_2, x2, batch = self.graph_pooling_2(
            x2, vertices_pool_1, batch=batch
        )

        x3 = self.gcn_3(x2, batch=batch)

        # Global feature generating B*1024
        feat_max = global_max_pool(x3, batch=batch)
        feat_avg = global_mean_pool(x3, batch=batch)
        feat_gf = torch.cat((feat_max, feat_avg), dim=1)
        # Decoder coarse input
        coarse_pcd = self.fc(feat_gf)

        coarse_pcd = rearrange(coarse_pcd, "b (d n)-> (b n) d", d=3)

        if return_batch:
            batch = torch.repeat_interleave(
                torch.arange(feat_max.shape[0]), self.num_pc
            ).to(device)
            return coarse_pcd, feat_max, batch
        else:
            return coarse_pcd, feat_max


class FbacBlock(nn.Module):
    def __init__(self, up_factor: int = 2):
        super().__init__()
        self.up_factor = up_factor

        # Feature extraction
        self.gcn = DynamicEdgeConv(nn=edge_conv_nn(3, 64), k=16)
        self.mlp = MLP([64 * 2, 128, 32])

        # Feedback exploitation
        self.cross_transformer = CrossTransformer(
            in_channel=32, pos_hidden_dim=32, attn_hidden_dim=32
        )

        # Node expansion
        self.nodeshuffle = upsample.NodeShuffle(32, 32, k=8, r=up_factor)

        # Coordinate generation
        self.mlp_delta = MLP([32, 64, 64, 3])

        self.up_sampler = nn.Upsample(scale_factor=up_factor)

    def forward(
        self,
        pcd,
        pcd_next=None,
        feat_next=None,
        batch_current=None,
        batch_next=None,
        return_batch: bool = True,
    ):
        # b, C, n_prev = pcd.shape

        # Step 1: Feature Extraction
        feat = self.gcn(pcd, batch=batch_current)
        feat = to_dense_batch(feat, batch=batch_current)[0]
        feat = torch.cat(
            [
                feat,
                repeat(
                    reduce(feat, "b n c -> b n", "max"),
                    "b n -> b n new_axis",
                    new_axis=feat.shape[-1],
                ),
            ],
            -1,
        )
        feat = rearrange(feat, "b n c -> (b n) c")
        feat = self.mlp(feat)

        # Step 2: Feedback Exploitation
        if pcd_next is None:
            pcd_next, feat_next, batch_next = pcd, feat, batch_current
        feat = self.cross_transformer(
            pcd,
            pcd_next,
            feat,
            feat_next,
            batch_x=batch_current,
            batch_y=batch_next,
        )

        # Step 3: Feature Expansion
        feat, batch = self.nodeshuffle(feat, batch=batch_current, return_batch=True)

        # Step 4: Coordinate Generation
        delta = self.mlp_delta(feat)
        u = repeat(pcd, "n c -> (n d) c", d=self.up_factor)
        pcd_child = u + delta

        if return_batch:
            return pcd_child, feat, batch
        else:
            return pcd_child, feat


class FeedbackRefinementNet(nn.Module):
    def __init__(
        self,
        up_factors=None,
        cycle_num=1,
        n_points_start=512,
        return_all: bool = False,
    ):
        super().__init__()
        self.return_all = return_all
        self.n_points_start = n_points_start
        self.cycle_num = cycle_num
        if up_factors is None:
            up_factors = [1]

        self.uppers = nn.ModuleList(
            [FbacBlock(up_factor=factor) for factor in up_factors]
        )

        self.flatten_batch = Rearrange("b n c -> (b n) c")

    def forward(self, pcd, partial):

        # Init input
        arr_pcd = []
        pcd = fps_subsample(
            torch.cat([pcd, partial], dim=1), self.n_points_start
        )  # [b n_start 3]

        feat_state = []
        pcd_state = []

        for cycle in range(self.cycle_num):
            pcd_list = []
            feat_list = []
            for upper_idx, upper in enumerate(self.uppers):
                # First timestep
                if cycle == 0:
                    # Add partial and fps only when they're available
                    if upper_idx > 0:
                        n_points = pcd.shape[1]
                        # Concatenate pcd and partial
                        pcd = torch.cat([pcd, partial], dim=1)
                        # Sample back with fps
                        pcd = fps_subsample(pcd, n_points)

                    batch_current = get_batch(pcd)
                    pcd, feat, b_ = upper(
                        self.flatten_batch(pcd), batch_current=batch_current
                    )
                    pcd = to_dense_batch(pcd, batch=b_)[0]

                # Next timesteps
                else:
                    pcd_next = pcd_state[cycle - 1][upper_idx]
                    feat_next = feat_state[cycle - 1][upper_idx]

                    # First fbac block
                    if upper_idx == 0:
                        pcd = pcd_state[cycle - 1][0]
                        pcd = torch.cat([pcd, partial], dim=1)
                        pcd = fps_subsample(pcd, self.n_points_start)
                    else:
                        pcd = pcd_list[upper_idx - 1]  # take last pcd
                        n_points = pcd_state[cycle - 1][upper_idx - 1].shape[1]
                        pcd = torch.cat([pcd, partial], dim=1)
                        pcd = fps_subsample(pcd, n_points)

                    batch_current = get_batch(pcd)
                    batch_next = get_batch(pcd_next)
                    pcd, feat, b_ = upper(
                        self.flatten_batch(pcd),
                        self.flatten_batch(pcd_next),
                        feat_next,
                        batch_current=batch_current,
                        batch_next=batch_next,
                    )
                    pcd = to_dense_batch(pcd, batch=b_)[0]

                pcd_list.append(pcd)
                feat_list.append(feat)

                if self.return_all:
                    arr_pcd.append(pcd)
                else:
                    if cycle == self.cycle_num - 1:
                        arr_pcd.append(pcd)

            # Saving present time step states
            pcd_state.append(pcd_list)
            feat_state.append(feat_list)
        return arr_pcd


class Model(nn.Module):
    def __init__(
        self,
        n_pc: int = 128,
        n_points_start: int = 512,
        up_factors: list = None,
        cycle_num: int = 3,
    ):
        super().__init__()

        self.coarse_net = HGNet(num_pc=n_pc)

        if up_factors is None:
            up_factors = [1, 2, 2]

        cycle_num = cycle_num
        self.refiner = FeedbackRefinementNet(
            up_factors=up_factors,
            cycle_num=cycle_num,
            return_all=True,
            n_points_start=n_points_start,
        )
        self.flatten_batch = Rearrange("b n c -> (b n) c")

    def forward(self, x):

        # Coarse generation
        coarse_pcd, _, batch = self.coarse_net(
            self.flatten_batch(x), batch=get_batch(x)
        )
        # feedback refinement stage
        coarse_pcd_dense = to_dense_batch(coarse_pcd, batch=batch)[0]
        res_pcds = self.refiner(coarse_pcd_dense, x)

        fine = res_pcds[-1]
        return coarse_pcd_dense, res_pcds, fine
