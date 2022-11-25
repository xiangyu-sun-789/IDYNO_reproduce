import os

import pandas as pd
from lingam.utils import make_dot
from sklearn import preprocessing

from causalnex.structure.transformers import DynamicDataTransformer
from idyno.locally_connected import LocallyConnected
from idyno.lbfgsb_scipy import LBFGSBScipy
from idyno.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math

"""
IDYNO consists of 3 MLPs:
    1. a MLP for instantaneous edges W. This should be identical to notears-MLP.
    2. a 2nd MLP for time-lagged edges A. This one does not need acyclicity constraint. 
        All the estimated edges in A points to instantaneous time.
    3. a 3rd MLP to concatenate the first 2 MLPs.
"""


class IDYNO_W(nn.Module):
    def __init__(self, dims, bias=True):
        super(IDYNO_W, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class IDYNO_A(nn.Module):
    def __init__(self, dims, p, bias=True):
        super(IDYNO_A, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.p = p
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d * self.p, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d * self.p, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d * self.p):
                    bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        return torch.tensor(0)

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d * self.p)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class MLP3(nn.Module):
    def __init__(self, dims, bias=True):
        super(MLP3, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.fc1 = nn.Linear(d * 2, d * dims[1], bias=bias)
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x1, x2):  # [n, d] -> [n, d]
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model_W, model_A, model_3, X, Xlags, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None

    parameters_W = model_W.parameters()
    parameters_A = model_A.parameters()
    parameters_3 = model_3.parameters()
    params = list(parameters_W) + list(parameters_A) + list(parameters_3)
    optimizer = LBFGSBScipy(params)

    X_torch = torch.from_numpy(X)
    Xlags_torch = torch.from_numpy(Xlags)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()

            X_hat_W = model_W(X_torch)
            X_hat_A = model_A(Xlags_torch)
            X_hat = model_3(X_hat_W, X_hat_A)

            loss = squared_loss(X_hat, X_torch)
            h_val = model_W.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * (model_W.l2_reg() + model_A.l2_reg())
            l1_reg = lambda1 * (model_W.fc1_l1_reg() + model_A.fc1_l1_reg())
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model_W.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def train_IDYNO(model_W: nn.Module,
                model_A: nn.Module,
                model_3: nn.Module,
                X: np.ndarray,
                Xlags: np.ndarray,
                lambda1: float = 0.,
                lambda2: float = 0.,
                max_iter: int = 100,
                h_tol: float = 1e-8,
                rho_max: float = 1e+16,
                w_threshold: float = 0.0):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model_W, model_A, model_3, X, Xlags, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model_W.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0

    A_est = model_A.fc1_to_adj()
    A_est[np.abs(A_est) < w_threshold] = 0

    return W_est, A_est


def draw_DAGs_using_LINGAM(file_name, adjacency_matrix, variable_names):
    # https://github.com/WillKoehrsen/Data-Analysis/issues/36#issuecomment-498710710

    # direction of the adjacency matrix needs to be transposed.
    # in LINGAM, the adjacency matrix is defined as column variable -> row variable
    # in NOTEARS, the W is defined as row variable -> column variable

    # the default value here was 0.01. Instead of not drawing edges smaller than 0.01, we eliminate edges
    # smaller than `w_threshold` from the estimated graph so that we can set the value here to 0.
    lower_limit = 0.0

    dot = make_dot(np.transpose(adjacency_matrix), labels=variable_names, lower_limit=lower_limit)

    dot.format = 'png'
    dot.render(file_name)


def main():
    result_folder = "./temp_result/"
    os.makedirs(result_folder, exist_ok=True)

    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import idyno.utils as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    p = 3

    variable_names_W = ['X{}'.format(j) for j in range(1, d + 1)]

    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt(result_folder + 'W_true.csv', B_true, delimiter=',')
    draw_DAGs_using_LINGAM(result_folder + "W_true", B_true, variable_names_W)

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt(result_folder + 'X.csv', X, delimiter=',')

    print("X.shape: ", X.shape)

    # normalize X
    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)

    normalized_data_df = pd.DataFrame(normalized_X, index=None, columns=variable_names_W)

    # concatenate data for time lags, copy from dynotears
    X, Xlags = DynamicDataTransformer(p=p).fit_transform(normalized_data_df, return_df=False)
    print("X.shape: ", X.shape)
    print("Xlags.shape: ", Xlags.shape)

    model_W = IDYNO_W(dims=[d, 10, 1], bias=True)
    model_A = IDYNO_A(dims=[d, 10, 1], p=p, bias=True)
    model_3 = MLP3(dims=[d, 10, 1], bias=True)
    w_threshold = 0.0
    W_est, A_est = train_IDYNO(model_W, model_A, model_3, X, Xlags, lambda1=0.01, lambda2=0.01, w_threshold=w_threshold)

    np.savetxt(result_folder + 'W_est.csv', W_est, delimiter=',')
    draw_DAGs_using_LINGAM(result_folder + "W_est", W_est, variable_names_W)

    variable_names_A = ['X{}_(t-{})'.format(j, k) if k != 0 else 'X{}_(t)'.format(j, k) for k in range(0, p + 1) for j
                        in range(1, d + 1)]
    print("variable_names_A: ", variable_names_A)

    A_est_full = np.zeros((d * (p + 1), d * (p + 1)))
    A_est_full[d:, :d] = A_est
    np.savetxt(result_folder + 'A_est_full.csv', A_est_full, delimiter=',')
    draw_DAGs_using_LINGAM(result_folder + "A_est", A_est_full, variable_names_A)

    assert ut.is_dag(W_est)

    # TODO:
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)


if __name__ == '__main__':
    main()
