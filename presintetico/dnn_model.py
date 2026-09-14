import torch
import torch.nn as nn


def emb_dim(cardinality):
    return int(
        min(
            50,
            max(
                2,
                round(
                    1.6
                    * cardinality ** 0.56
                ),
            ),
        )
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
    ):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

        if in_dim == out_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(
                in_dim,
                out_dim,
                bias=False,
            )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)

        x = self.main(x)
        x = x + residual
        x = self.activation(x)
        x = self.dropout(x)

        return x


class TabularDNN(nn.Module):
    def __init__(
        self,
        n_numeric,
        cat_cardinalities,
        hidden,
        n_classes,
        dropout,
    ):
        super().__init__()

        # ====================================================
        # EMBEDDINGS
        # ====================================================

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                cardinality,
                emb_dim(cardinality),
            )
            for cardinality
            in cat_cardinalities
        ])

        emb_total = sum(
            emb_dim(cardinality)
            for cardinality
            in cat_cardinalities
        )

        in_dim = (
            n_numeric
            + emb_total
        )

        # ====================================================
        # NORMALIZACION DE ENTRADA NUMERICA
        # ====================================================

        self.numeric_bn = (
            nn.BatchNorm1d(n_numeric)
            if n_numeric > 0
            else None
        )

        # ====================================================
        # MLP RESIDUAL
        # ====================================================

        blocks = []

        prev = in_dim

        for h in hidden:
            blocks.append(
                ResidualBlock(
                    prev,
                    h,
                    dropout,
                )
            )

            prev = h

        self.mlp = nn.Sequential(
            *blocks
        )

        # ====================================================
        # CLASIFICADOR
        # ====================================================

        self.output = nn.Linear(
            prev,
            n_classes,
        )

    def forward(
        self,
        x_num,
        x_cat,
    ):
        # x_cat tiene que ser torch.long

        if self.numeric_bn is not None:
            x_num = self.numeric_bn(
                x_num
            )

        embs = [
            emb(
                x_cat[:, i]
            )
            for i, emb
            in enumerate(
                self.embeddings
            )
        ]

        x = torch.cat(
            [x_num] + embs,
            dim=1,
        )

        x = self.mlp(x)

        return self.output(x)