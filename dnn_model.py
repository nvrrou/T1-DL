import torch
import torch.nn as nn

def emb_dim(cardinality):
    return int(min(50, max(2, round(1.6 * cardinality ** 0.56))))

class TabularDNN(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, hidden, n_classes, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim(card)) for card in cat_cardinalities
        ])
        emb_total = sum(emb_dim(c) for c in cat_cardinalities)
        in_dim = n_numeric + emb_total

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)
