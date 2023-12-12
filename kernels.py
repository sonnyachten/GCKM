import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from dataset_loader import DataLoader

Tensor = torch.Tensor


def kernel_factory(type: str, param: float = None, param2: int = None, **kwargs):
    if type == 'rbf':
        kernel = GaussianKernelTorch(sigma2=param)
    elif type == 'laplace':
        kernel = LaplaceKernelTorch(sigma=param)
    elif type == 'linear':
        kernel = LinearKernel()
    elif type == 'poly':
        kernel = PolyKernelTorch(d=param2, t=param)
    elif type == 'npoly':
        kernel = NormPolyKernelTorch(d=param2, t=param)
    elif type == 'rbf_sparse':
        kernel = GaussianKernelTorchSparse(sigma2=param)
    elif type == 'wwl':
        kernel = WwlKernel(param, param2, kwargs['h'])
    else:
        ValueError('Unknown kernel function')
    return kernel


class WwlKernel(nn.Module):
    def __init__(self, dataset, gamma=1.0, h=2, discrete=False):
        super().__init__()
        import ot
        data, _, _ = DataLoader(dataset, preprocess='standardize')
        data.edge_index = to_undirected(data.edge_index)
        adj_mat = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                               sparse_sizes=(data.x.shape[0], data.x.shape[0])).to_dense().numpy()
        wl_embeddings = self.create_labels_seq_cont(data.x.numpy(), adj_mat, h)
        ground_distance = 'hamming' if discrete else 'euclidean'
        M = ot.dist(wl_embeddings, wl_embeddings, metric=ground_distance)

        self.K = torch.exp(-gamma * torch.Tensor(M))

    def create_labels_seq_cont(self, node_feat, adj_mat, h):
        node_feat = [node_feat]
        adj_cur = adj_mat + np.identity(adj_mat.shape[0], dtype='int32')
        adj_cur = self.create_adj_avg(adj_cur)

        for it in range(1, h + 1):
            np.fill_diagonal(adj_cur, 0)
            node_feat_cur = 0.5 * (np.dot(adj_cur, node_feat[it - 1]) + node_feat[it - 1])
            node_feat.append(node_feat_cur)

        return np.concatenate(node_feat, 1)

    def create_adj_avg(self, adj_cur):
        '''
        create adjacency
        '''
        deg = np.sum(adj_cur, axis=1)
        deg = np.asarray(deg).reshape(-1)

        deg[deg != 1] -= 1

        deg = 1 / deg

        # deg = deg.astype(np.float32)

        deg_mat = np.diag(deg)
        # adj_cur = adj_cur.dot(deg_mat.T).T
        adj_cur = np.matmul(deg_mat, adj_cur.T, dtype=np.float32)

        return adj_cur

    def forward(self, X=None, Y=None) -> Tensor:
        return self.K


class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x n matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: n x M kernel matrix
        """
        if Y is None:
            Y = X

        return torch.mm(X.t(), Y)


class NormPolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=10.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        X = X.t()
        Y = Y.t()
        D1 = torch.diag(1. / torch.sqrt(torch.pow(torch.sum(torch.pow(X, 2), dim=1) + self.c ** 2, self.d)))
        D2 = torch.diag(1. / torch.sqrt(torch.pow(torch.sum(torch.pow(Y, 2), dim=1) + self.c ** 2, self.d)))
        return torch.matmul(torch.matmul(D1, torch.pow((torch.mm(X, Y.t()) + self.c ** 2), self.d)), D2)


class PolyKernelTorch(nn.Module):
    def __init__(self, d: int, t=10.0) -> None:
        super().__init__()
        self.d = d
        self.c = t

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x N matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: N x M kernel matrix
        """
        if Y is None:
            Y = X
        X = X.t()
        Y = Y.t()

        return torch.pow((torch.mm(X, Y.t()) + self.c ** 2), self.d)


class GaussianKernelTorchSparse(nn.Module):
    def __init__(self, sigma2=0.0):
        super(GaussianKernelTorchSparse, self).__init__()
        self.batch_size = 1000
        self.kernel_dense = GaussianKernelTorch(sigma2)

    def _threshold(self, A, percentage=0.01):
        # compute the maximum and minimum entries
        A_max = torch.max(A)
        A_min = torch.min(A)

        # assume that the maximum and minimum values correspond to the mean plus and minus 2 standard deviations, respectively
        mean = (A_max + A_min) / 2
        std = (A_max - A_min) / 4.0

        # compute the threshold using the desired percentage
        threshold = A_min + (A_max - A_min) * percentage

        # set the entries below the threshold to zero
        A[A < threshold] = 0
        return A

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x n matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: n x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        X = X.t()
        Y = Y.t()

        results = []
        for i in range(0, N, self.batch_size):
            # extract a chunk of X of size batch_size
            X_batch = X[i:i + self.batch_size]

            # apply f to X_batch and Y
            result_batch = self.kernel_dense(X_batch.t(), Y.t())

            # sparsify
            # result_batch[np.isclose(result_batch, 0)] = 0
            result_batch = self._threshold(result_batch)
            print(f"Number of zero elements is {np.sum(np.isclose(result_batch, 0))}")
            result_batch = result_batch.to_sparse()

            # append the result to the list of results
            results.append(result_batch)

        # concatenate the results to obtain the full matrix
        C = torch.cat(results, dim=0)

        return C.to_dense()


class GaussianKernelTorch(nn.Module):
    def __init__(self, sigma2=0.0):
        super(GaussianKernelTorch, self).__init__()
        self.sigma2 = torch.tensor([float(sigma2)])
        # self.sigma2 = Parameter(torch.tensor([float(sigma2)]), requires_grad=False)
        # self.register_parameter("sigma2", self.sigma2)

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x n matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: n x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes adj matrix of the squared norm of the difference.
            """
            x1 = torch.t(x1)
            x2 = torch.t(x2)
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
            res = res.clamp_min_(1e-30).sqrt_()
            return res

        D = my_cdist(X, Y)

        return torch.exp(- torch.pow(D.to(X.device), 2) / (2 * self.sigma2.to(X.device)))


class LaplaceKernelTorch(nn.Module):
    def __init__(self, sigma=50.0):
        super(LaplaceKernelTorch, self).__init__()
        self.sigma = torch.tensor([float(sigma)])
        # self.sigma2 = Parameter(torch.tensor([float(sigma2)]), requires_grad=False)
        # self.register_parameter("sigma2", self.sigma2)

    def forward(self, X: Tensor, Y: Tensor = None) -> Tensor:
        """
        Computes the kernel matrix for some observation matrices X and Y.
        :param X: d x n matrix
        :param Y: d x M matrix. If not specified, it is assumed to be X.
        :return: n x M kernel matrix
        """
        if Y is None:
            Y = X
        N = X.shape[1] if len(X.shape) > 1 else 1
        M = Y.shape[1] if len(Y.shape) > 1 else 1

        def my_cdist(x1, x2):
            """
            Computes adj matrix of the l1 norm of the difference.
            """
            raise NotImplementedError('Laplace kernel not implemented')

        D = my_cdist(X, Y)

        return torch.exp(-D.to(self.sigma.device) / (2 * self.sigma))
