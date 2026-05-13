import torch
import torch.nn as nn
from src.model.base import BaseNet


class MLP(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.head = nn.ModuleList()
        self.regression = getattr(cfg, "regression", True)
        self.embed_dim = cfg.embed_dim

        dropout_layer = getattr(cfg, "dropout_layer", [])
        norm_layer = getattr(cfg, "norm_layer", [])
        dropout_rate = getattr(cfg, "dropout_rate", 0.0)
        activation = getattr(cfg, "activation", "ReLU")
        self.token_avg = getattr(cfg, "token_avg", False)
        activation_final = getattr(cfg, "activation_final", True)
        self.num_classes = getattr(cfg, "num_classes", 1)

        input_dim = (
            cfg.embed_dim
            if getattr(cfg, "token_avg", False)
            else cfg.num_tokens * cfg.embed_dim
        )

        if cfg.num_layers == 1:
            self.head.append(nn.Linear(input_dim, self.num_classes))
        else:
            assert len(cfg.hidden_sizes) == cfg.num_layers - 1

            self.head.append(nn.Linear(input_dim, cfg.hidden_sizes[0]))
            if not activation_final:
                self.head.append(getattr(nn, activation)())

            if 0 in norm_layer:
                self.head.append(nn.BatchNorm1d(cfg.hidden_sizes[0]))
            if 0 in dropout_layer:
                self.head.append(nn.Dropout(dropout_rate))

            if activation_final:
                self.head.append(getattr(nn, activation)())

            for i in range(1, cfg.num_layers - 1):
                self.head.append(
                    nn.Linear(cfg.hidden_sizes[i - 1], cfg.hidden_sizes[i])
                )
                if not activation_final:
                    self.head.append(getattr(nn, activation)())
                if i in norm_layer:
                    self.head.append(nn.BatchNorm1d(cfg.hidden_sizes[i]))
                if i in dropout_layer:
                    self.head.append(nn.Dropout(dropout_rate))
                if activation_final:
                    self.head.append(getattr(nn, activation)())
            self.head.append(nn.Linear(cfg.hidden_sizes[-1], self.num_classes))

        self.head = self.head.to(dtype=self.dtype["head"])

    def forward(self, x):
        use_cuda = torch.cuda.is_available()

        if self.backbone_name is not None:
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=self.dtype["backbone"]):
                    x = (
                        self.batchseq2embed(x)
                        if not self.regression
                        else self.seq2embed(x)
                    )
            else:
                # CPU path without autocast
                x = self.batchseq2embed(x) if not self.regression else self.seq2embed(x)

        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=self.dtype["head"]):
                if self.token_avg:
                    x = x.mean(dim=1)  # (bs, L, D) -> (bs, D)
                else:
                    x = x.view(x.size(0), -1)  # (bs, L, D) -> (bs, L*D)

                for layer in self.head:
                    x = layer(x)
        else:
            # CPU path without autocast
            if self.token_avg:
                x = x.mean(dim=1)  # (bs, L, D) -> (bs, D)
            else:
                x = x.view(x.size(0), -1)  # (bs, L, D) -> (bs, L*D)

            for layer in self.head:
                x = layer(x)

        if not self.regression and self.num_classes == 1:
            x = torch.sigmoid(x)

        return x.squeeze()
