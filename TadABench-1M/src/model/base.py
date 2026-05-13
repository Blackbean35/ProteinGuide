import torch
import torch.nn as nn
from typing import List, Tuple
from src.backbone.constant import MODEL_MAP
from src.dataset import modality_map


MODULES = ["backbone", "head", "other"]


class BaseNet(nn.Module):
    DTYPES = dict(fp32=torch.float32, bf16=torch.bfloat16)

    def __init__(self, cfg):
        super(BaseNet, self).__init__()
        self.dtype = {module: self.DTYPES[dtype] for module, dtype in cfg.dtype.items()}
        self.frozen_backbone = getattr(cfg, "frozen_backbone", False)
        self.seq_type = cfg.seq_type

        self.backbone_name = None if cfg.use_embed_mapper else cfg.embed_name
        if self.backbone_name is not None:
            self.init_backbone(cfg)

    def init_backbone(self, cfg):
        self.backbone = MODEL_MAP(self.backbone_name, cfg.seq_type, cfg)

        for param in self.backbone.model.parameters():
            param.requires_grad = not self.frozen_backbone

    def batchseq2seqs(self, seqs: Tuple[List[str]]):
        seqs_batch = [[] for _ in range(len(seqs[1]))]
        for ki in seqs:
            for b, seq in enumerate(ki):
                seqs_batch[b].append(seq)
        seqs_view = [
            modality_map(self.seq_type, seq) for batch in seqs_batch for seq in batch
        ]
        return seqs_view

    def seq2embed(self, seqs: List[str]):
        if self.frozen_backbone:
            with torch.no_grad():
                x = self.backbone(seqs)
        else:
            x = self.backbone(seqs)
        return x

    def batchseq2embed(self, seqs: List[str]):
        bs = len(seqs[1])
        seqs_view = self.batchseq2seqs(seqs)
        if self.frozen_backbone:
            with torch.no_grad():
                x = self.backbone(seqs_view)
        else:
            x = self.backbone(seqs_view)
        x = x.reshape(bs, x.shape[0] // bs, x.shape[1], x.shape[2])
        return x

    def forward(self, x, batch: bool = True):
        if self.backbone is not None:
            if batch:
                x = self.batchseq2embed(
                    x
                )  # x: List[Tuple[str]], (k, bs, L) -> torch.Tensor(bs, k, L, D)
            else:
                x = self.seq2embed(x)  # x: List[str], (bs, L) -> torch.Tensor(bs, L, D)
        return x
