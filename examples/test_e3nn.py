"""Different paramters for the different atom types

based on `tetris_polynomial`

idea:
if we have num_z types of atoms we have num_z^2 types of edges.
Instead of having spherical harmonics for the edge attributes
we have num_z^2 times the spherical harmonics, all zero except for the type of the edge.

>>> test()
"""
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct


class InvariantPolynomial(torch.nn.Module):
    def __init__(self, irreps_out, num_z, lmax) -> None:
        super().__init__()
        self.num_z = num_z

        # spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # to MULTIPLY the edge type one-hot with the spherical harmonics to get the edge attributes
        self.mul = TensorProduct(
            [(num_z**2, "0e")],  # edge type one-hot
            self.irreps_sh, 
            [(num_z**2, ir) for _, ir in self.irreps_sh], # edge attributes
            [(0, l, l, "uvu", False) for l in range(lmax + 1)], # uvu means the output is symmetric under the exchange of the two inputs
        )
        irreps_attr = self.mul.irreps_out

        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o") # these mean the output of the model
        irreps_out = o3.Irreps(irreps_out)

        # tp means tensor product
        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=irreps_attr,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=irreps_attr,
            irreps_out=irreps_out,
        )

    def forward(self, data) -> torch.Tensor:
        num_neighbors = 3  # typical number of neighbors
        num_nodes = 4  # typical number of nodes
        num_z = self.num_z  # number of atom types

        # graph
        edge_src, edge_dst = radius_graph(data.pos, 10.0, data.batch) # cutoff here

        # spherical harmonics
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=False, normalization="component")

        # edge types
        edge_zz = num_z * data.z[edge_src] + data.z[edge_dst]  # from 0 to num_z^2 - 1
        edge_zz = torch.nn.functional.one_hot(edge_zz, num_z**2).mul(num_z)
        edge_zz = edge_zz.to(edge_sh.dtype)

        # edge attributes
        edge_attr = self.mul(edge_zz, edge_sh)

        # For each node, the initial features are the sum of the spherical harmonics of the neighbors
        node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(node_features[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp2(node_features[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each graph, all the node's features are summed
        return scatter(node_features, data.batch, dim=0).div(num_nodes**0.5)



def compute_local_environment(pos, z, radius=1.5):
    """
    Compute the local environment for each atom, which includes the relative positions
    of its neighboring atoms within a given radius and the types of neighboring atoms.

    Parameters:
    pos (torch.Tensor): Tensor of shape (N, 3) or (F, N, 3) containing the positions of N atoms
                        or F frames of N atoms.
    z (torch.Tensor): Tensor of shape (N,) or (F, N) containing the types of N atoms or F frames of N atoms.
    radius (float): The radius within which to consider neighboring atoms.

    Returns:
    tuple: Two tensors, one for the relative positions and one for the neighbor types.
           If input is (N, 3), returns tensors of shape (1, N, N-1, 3) and (1, N, N-1, 1).
           If input is (F, N, 3), returns tensors of shape (F, N, N-1, 3) and (F, N, N-1, 1).
    """
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)  # Add a frame dimension if not present
        z = z.unsqueeze(0)  # Add a frame dimension if not present

    all_relative_pos = []
    all_neighbor_types = []

    for frame_pos, frame_z in zip(pos, z):
        # Create a Data object
        data = Data(pos=frame_pos, z=frame_z)

        # Compute the radius graph
        edge_index = radius_graph(data.pos, r=radius)

        # Initialize lists to store relative positions and neighbor types
        relative_pos = torch.zeros((frame_pos.size(0), frame_pos.size(0) - 1, 3))
        neighbor_types = torch.zeros((frame_pos.size(0), frame_pos.size(0) - 1, 1))

        # Iterate over the edges to compute relative positions and neighbor types
        for src, dst in edge_index.t().tolist():
            relative_pos[src, dst % (frame_pos.size(0) - 1)] = data.pos[dst] - data.pos[src]
            neighbor_types[src, dst % (frame_pos.size(0) - 1)] = data.z[dst]

        all_relative_pos.append(relative_pos)
        all_neighbor_types.append(neighbor_types)

    # Stack the lists to create tensors of shape (F, N, N-1, 3) and (F, N, N-1, 1)
    all_relative_pos = torch.stack(all_relative_pos)
    all_neighbor_types = torch.stack(all_neighbor_types)
    # convert all_neighbor_types to int
    all_neighbor_types = all_neighbor_types.to(torch.int64)

    if pos.size(0) == 1:
        return all_relative_pos[0], all_neighbor_types[0]  # Return single tensors if there was only one frame
    return all_relative_pos, all_neighbor_types




if __name__ == "__main__":
    
    torch.set_default_dtype(torch.float64)

    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.5],
        ]
    )

    # atom type
    z = torch.tensor([0, 1, 2, 2])

    dataset = [Data(pos=pos @ R.T, z=z) for R in o3.rand_matrix(10)]
    data = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    print(f'dataset: {dataset}')

    f = InvariantPolynomial("0e+0o", num_z=3, lmax=3)

    out = f(data)

    # expect invariant output
    assert out.std(0).max() < 1e-5
    
    # print the output
    print(out)