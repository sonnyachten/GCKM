import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from definitions import TensorType
from kernels import kernel_factory
from utils import device


def orto(h, full=False):
    o = torch.mm(h.t(), h)
    o = o - torch.eye(*o.shape, device=o.device)
    n = torch.norm(o, "fro")
    return torch.pow(n, 2), n, None if not full else o


def zerosum(h, full=False):
    s = torch.sum(h, dim=0)
    n = torch.norm(s)
    return torch.pow(n, 2), n, None if not full else s


class noaggr(nn.Module):
    def __init__(self, x=None, e=None, cfg=None, layerwisein=True):
        super().__init__()

    def forward(self, x, edge_index=None):
        return x


class sumaggr(MessagePassing):
    def __init__(self, x=None, e=None, cfg=None, layerwisein=True):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j


class meanaggr(MessagePassing):
    def __init__(self, x=None, e=None, cfg=None, layerwisein=True):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        _, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class gcn_aggr(MessagePassing):
    def __init__(self, x=None, e=None, cfg=None, layerwisein=True):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCKMLayer(nn.Module):
    def __init__(self, x, edge_index, cfg, layerwisein, svd_type=None, svd_tol=None):
        super(GCKMLayer, self).__init__()
        xtrain = x
        self.s = cfg.s
        self.k = kernel_factory(cfg.kernel, cfg.kernelparam, cfg.kernelparam2)
        self.gamma = cfg.gamma
        self.n = xtrain.shape[0]
        self.register_parameter("h_st", Parameter(torch.randn((self.s, self.n), device=device), requires_grad=False))
        self.aggr = \
            {"mean": meanaggr, "sum": sumaggr, "no_aggr": noaggr, "gcn": gcn_aggr}[cfg.aggr.type](x=xtrain,
                                                                                                  e=edge_index,
                                                                                                  cfg=cfg.aggr,
                                                                                                  layerwisein=layerwisein)
        if layerwisein:
            assert svd_tol is not None and svd_type is not None
            self.layerwise_init(x, edge_index, svd_type, svd_tol)

    def forward(self, x=None, edge_index=None):
        return self.h_st.t()

    def loss(self, x, edge_index):
        a = self.aggr(x, edge_index)
        k = self.k(a.t()).to(device)
        mc = torch.eye(self.n, device=device) - torch.ones(self.n, self.n, device=device) / self.n
        l1 = - 1 / 2 * torch.trace(self.h_st @ mc @ k @ mc @ self.h_st.t())
        l1 = self.gamma * l1 / self.n
        ort1 = orto(self.h_st.t())[0] / self.n
        return l1, ort1

    def layerwise_init(self, x, edge_index, svd_type, svd_tol=None):
        x = x.type(TensorType)
        a = self.aggr(x, edge_index)
        k = self.k(a.t()).to(device)
        m = k - torch.sum(k, dim=0) / self.n
        if svd_type == 'lanczos':
            from scipy.sparse.linalg import eigsh
            _, h = eigsh(m.cpu().numpy(), k=self.s, tol=svd_tol)
            h = torch.from_numpy(h).to(device)
        else:
            h, _, _ = torch.svd(m, some=False)
        h = h.to(device)

        h = h[:, :self.s]
        self.h_st[:] = h.t()

    def predict(self, x=None, e=None):
        return self.h_st.t()


class SemiSupRkmLayer(nn.Module):
    def __init__(self, cfg, x, l, c, layerwisein):
        super(SemiSupRkmLayer, self).__init__()
        self.eta = cfg.eta
        self.lam1 = cfg.lam1
        self.lam2 = cfg.lam2
        self.gamma = cfg.gamma
        self.p = c.shape[1]
        self.n = x.shape[0]
        self.l = l
        self.c = c  # mask validation and test labels
        self.k1 = kernel_factory(cfg.kernel1.type, cfg.kernel1.param, cfg.kernel1.param2)
        if cfg.kernel2.type == 'wwl':
            self.multiview = True
            self.kappa = cfg.kernel2.kappa
            self.rho = cfg.kernel2.rho
            self.k2 = kernel_factory(cfg.kernel2.type, cfg.kernel2.dataset, cfg.kernel2.param1, h=cfg.kernel2.param2)
        elif cfg.kernel2.type in ['zero', 'no', 'none', None]:
            self.multiview = False
        else:
            raise AssertionError
        self.register_parameter("h_class", Parameter(torch.randn((self.p, self.n), device=device), requires_grad=False))
        self.r = torch.randn(self.n)
        self.bias = torch.randn(self.p)
        if layerwisein:
            self.solve_and_update(x)

    def forward(self, x=None):
        return self.h_class.t()

    def loss(self, x):
        xtr = x
        if self.multiview:
            k = (1 - self.rho) * (
                    (1 - self.kappa) * self.k1(xtr.t()).to(device) + self.kappa * self.k2(xtr.t()).to(device)
            ) + self.rho * self.k1(xtr.t()).to(device) * self.k2(xtr.t()).to(device)
        else:
            k = self.k1(xtr.t()).to(device)

        d = torch.sum(k, 1)
        dinv = torch.pow(d, -1)
        r = dinv / self.lam1 - self.l / self.lam2
        rmat = torch.diagflat(r)
        loss = - 1 / 2 * torch.trace(self.h_class @ rmat @ k @ rmat @ self.h_class.t()) \
               + 1 / 2 * torch.trace(self.h_class @ rmat @ self.h_class.t()) \
               - 1 / self.lam2 * torch.trace(self.h_class @ self.c)

        zs = zerosum(r @ self.h_class.t())[0] / self.n
        return self.gamma * loss / self.n, zs

    def solve_and_update(self, x):
        xtr = x
        if self.multiview:
            k = (1 - self.rho) * (
                    (1 - self.kappa) * self.k1(xtr.t()).to(device) + self.kappa * self.k2(xtr.t()).to(device)
            ) + self.rho * self.k1(xtr.t()).to(device) * self.k2(xtr.t()).to(device)
        else:
            k = self.k1(xtr.t()).to(device)
        d = torch.sum(k, 1)
        if torch.min(torch.abs(d)) < 1e-32:
            raise ValueError('kernel matrix has at least one zero degree')
        dinv = torch.pow(d, -1)
        self.r = dinv / self.lam1 - self.l / self.lam2
        if torch.min(torch.abs(self.r)) < 1e-32:
            raise ValueError('r has at least one zero value')
        s = - self.r.expand(self.n, -1) / torch.sum(self.r)
        s[range(self.n), range(self.n)] = torch.diag(s) + 1

        b = ((k.t() * self.r.view(-1, 1)).t()) * self.r.view(-1, 1)
        left = k.add_(-k @ self.r / torch.sum(self.r))
        left = left.t().mul_(self.r.unsqueeze(1)).t()
        left.mul_(- 1 / self.eta)
        left[range(self.n), range(self.n)] += 1
        left.mul_(self.r.unsqueeze(1))
        right = 1 / self.lam2 * s.t() @ self.c

        left[np.isclose(left.cpu().numpy(), 0)] = 0

        h = torch.linalg.solve(left, right)

        self.h_class[:] = h.t()
        self.bias = - 1 / torch.sum(self.r) * (
                1 / self.eta * torch.ones(1, self.n, device=device) @ b @ self.h_class.t()
                +
                1 / self.lam2 * torch.ones(1, self.n, device=device) @ self.c
        )

    def predict(self, x=None):
        rinv = torch.diagflat(torch.pow(self.r, -1))
        e_tr = self.h_class.t() - rinv @ self.c / self.lam2
        return e_tr


class DeepGCKM(nn.Module):
    def __init__(self, cfg, edge_index, x, l, c, codes, trainmask=None, valmask=None, testmask=None, y=None):
        super(DeepGCKM, self).__init__()

        self.edge_index = edge_index
        self.x = x
        self.n = x.shape[0]
        self.p = c.shape[1]
        self.codes = codes
        self.mean_codes = torch.mean(codes, dim=0)
        self.centered_codes = self.codes - self.mean_codes
        self.num_gckm_layers = sum(["gckm_layer" in x for x in cfg.keys()])
        self.init_performance = {'train_acc': None, 'val_acc': None, 'cos_sim': None, 'comb_metric': None,
                                 'test_acc': None}
        self.gckm_layers = []
        for i in range(1, self.num_gckm_layers + 1):
            cfg_layer = cfg[f"gckm_layer{i}"]
            layer = GCKMLayer(x, edge_index, cfg_layer, cfg.layerwisein, svd_type=cfg.svd_type, svd_tol=cfg.svd_tol)
            self.gckm_layers.append(layer)
            x = layer.predict(x, edge_index)

        self.semisup_layer = SemiSupRkmLayer(cfg.SemiSup_layer, x, l, c, cfg.layerwisein)
        self.init_performance = self.acc_dict(y, trainmask, valmask, testmask, x, init=True)

    def forward(self):
        pass

    def loss(self):
        tot_ener = 0
        tot_orto = 0
        x = self.x
        for layer in self.gckm_layers:
            ener, orto = layer.loss(x, self.edge_index)
            x = layer.predict(x, self.edge_index)
            tot_ener, tot_orto = tot_ener + ener, tot_orto + orto
        ener, zs = self.semisup_layer.loss(x)
        tot_ener = tot_ener + ener

        return tot_ener, tot_orto / self.num_gckm_layers, zs

    def predict(self):
        x = self.x
        for level in self.gckm_layers:
            x = level.predict(x, self.edge_index)
        e = self.semisup_layer.predict(x)
        y_pred = torch.argmax(e, dim=1)  # only works for 1_vs_all encoding
        return y_pred, e

    def acc_dict(self, y, trainmask, valmask, testmask=None, x=None, init=False):
        if init:
            e = self.semisup_layer.predict(x)
            y_pred = torch.argmax(e, dim=1)
        else:
            y_pred, e = self.predict()
        testmask = torch.nonzero(testmask).flatten()
        num_test = testmask.shape[0]
        acc_train = 100 * accuracy_score(y_pred[trainmask].cpu().numpy(), y[trainmask].cpu().numpy())
        acc_test = 100 * accuracy_score(y_pred[testmask].cpu().numpy(), y[testmask].cpu().numpy())

        cos_sim = 0
        num_slices = num_test // 1000
        for testslice in range(num_slices + 1):
            if testslice < num_slices:
                mask = testmask[testslice:testslice + 1000]
            elif testslice == num_slices:
                mask = testmask[testslice * 1000:]
            ec = e[mask] - self.mean_codes
            dcos = torch.ones((mask.shape[0], self.codes.shape[0]), device=device) - ec @ self.centered_codes.t() \
                   / torch.outer(torch.sqrt(torch.diag(ec @ ec.t())),
                                 torch.sqrt(torch.diag(self.centered_codes @ torch.t(self.centered_codes))))
            cos_sim += mask.shape[0] * 100 - 100 * torch.sum(torch.min(dcos, dim=1).values)

        cos_sim = cos_sim / num_test

        metrics = {"train_acc": acc_train, "test_acc": acc_test, "cos_sim": float(cos_sim)}

        if type(valmask) is dict:
            for mask_name in valmask:
                mask = valmask[mask_name]
                num_val = torch.count_nonzero(mask)
                acc_val = 100 * accuracy_score(y_pred[mask].cpu().numpy(), y[mask].cpu().numpy())
                combined_metric = (acc_val * num_val + cos_sim * num_test) / (num_val + num_test)
                metrics.update({"val_acc_" + mask_name: acc_val, "comb_metric_" + mask_name: float(combined_metric)})
            metrics.update({"val_acc": acc_val, "comb_metric": float(combined_metric)})
        elif type(valmask) is torch.Tensor:
            num_val = torch.count_nonzero(valmask)
            acc_val = 100 * accuracy_score(y_pred[valmask].cpu().numpy(), y[valmask].cpu().numpy())
            combined_metric = (acc_val * num_val + cos_sim * num_test) / (num_val + num_test)
            metrics.update({"val_acc": acc_val, "comb_metric": float(combined_metric)})
        else:
            raise TypeError('Validation mask should be a Tensor or a dict containing Tensors')
        return metrics

    def solve_and_update(self, x=None):
        if x is None:
            self.semisup_layer.solve_and_update(self.gckm_layers[-1](), None)
        else:
            raise NotImplementedError
