import torch
from torch import nn


class WeightOffsets(nn.Module):
    def __init__(self, row_dim, column_dim):
        super().__init__()
        self.v = nn.Parameter(torch.ones(1))
        self.linear1 = nn.Linear(1, row_dim)
        self.linear2 = nn.Linear(1, column_dim)
        self.linear_column = nn.Linear(row_dim, row_dim)
        self.linear_row = nn.Linear(column_dim, column_dim)

    def forward(self):
        vx = self.linear1(self.v) # (row_dim)
        vy = self.linear2(self.v) # (column_dim)
        # matrix multiplication -> (row_dim, column_dim)
        v_matrix = vx.unsqueeze(0).T * vy.unsqueeze(0)
        # columnwise
        v_matrix = self.linear_column(v_matrix.T)
        # rowwise
        v_matrix = self.linear_row(v_matrix.T)
        return v_matrix.T


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wo = WeightOffsets(32, 16)
        self.linear = nn.Linear(32, 16)
        self.init_weight = None
        self.wo_out = None
        self.linear.weight.register_hook(self.wo_backward)

    def wo_backward(self, grad):
        print("grad:", grad)
        grad = grad * self.init_weight
        self.wo_out.backward(grad)

    def update_weight(self):
        if self.init_weight is None:
            self.init_weight = self.linear.weight.data.clone()
        self.wo_out = self.wo()
        self.linear.weight.data = self.init_weight * (1 + self.wo_out)

    def forward(self, x):
        self.update_weight()
        y = self.linear(x)
        return y


if __name__ == '__main__':
    model = Model()
    # model = WeightOffsets(32, 16)
    # linear = torch.nn.Linear(32, 16)
    # # linear.requires_grad_(False)
    # init_weight = linear.weight.data.clone()
    optimizer = torch.optim.AdamW(model.wo.parameters(), lr=0.01)
    # train!
    model.train()
    optimizer.zero_grad()

    x = torch.randn(2, 32)
    y = torch.randn(2, 16)
    # wo_weight = model()
    print(model.wo.v)
    # linear.weight.data = init_weight * (1 + wo_weight)
    # out = linear(x)
    out = model(x)
    loss = nn.functional.mse_loss(y, out)
    # loss = wo_weight.sum()
    print("loss:", loss)
    loss.backward()
    # grad = linear.weight.grad * init_weight
    # wo_weight.backward(grad)
    optimizer.step()
    print(model.wo.v)