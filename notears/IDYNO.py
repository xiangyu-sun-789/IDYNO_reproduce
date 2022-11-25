import os

import pandas as pd
from lingam.utils import make_dot
from sklearn import preprocessing

from causalnex.structure.transformers import DynamicDataTransformer
from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math

"""
IDYNO consists of 3 MLPs:
    1. a MLP for instantaneous edges W. This should be identical to notears-MLP?
    2. a 2nd MLP for time-lagged edges A. This one does not need acyclicity constraint. 
        But need the constraint to allow edges only pointing to the current time.
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


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def train_IDYNO(model: nn.Module,
                X: np.ndarray,
                lambda1: float = 0.,
                lambda2: float = 0.,
                max_iter: int = 100,
                h_tol: float = 1e-8,
                rho_max: float = 1e+16,
                w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


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

    import notears.utils as ut
    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    p = 3

    variable_names = ['X{}'.format(j) for j in range(1, d + 1)]

    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt(result_folder + 'W_true.csv', B_true, delimiter=',')
    draw_DAGs_using_LINGAM(result_folder + "W_true", B_true, variable_names)

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt(result_folder + 'X.csv', X, delimiter=',')

    print("X.shape: ", X.shape)

    # normalize X
    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)

    normalized_data_df = pd.DataFrame(normalized_X, index=None, columns=variable_names)

    # concatenate data for time lags, copy from dynotears
    X, Xlags = DynamicDataTransformer(p=p).fit_transform(normalized_data_df, return_df=False)
    print("X.shape: ", X.shape)
    print("Xlags.shape: ", Xlags.shape)

    model = IDYNO_W(dims=[d, 10, 1], bias=True)
    W_est = train_IDYNO(model, X, lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt(result_folder + 'W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)

    draw_DAGs_using_LINGAM(result_folder + "W_est", W_est, variable_names)


if __name__ == '__main__':
    main()
