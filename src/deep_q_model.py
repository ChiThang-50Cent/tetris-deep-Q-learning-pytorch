import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.ln2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.ln3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)

        return x 