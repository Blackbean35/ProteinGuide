import torch
import torch.nn as nn
from typing import List
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESM3Model(nn.Module):
    model_names = ["esm3_sm_open_v1", "esmc_300m", "esmc_600m"]

    def __init__(self, model_name: str, del_special_tokens: bool) -> None:
        super().__init__()

        assert model_name in self.model_names
        self.model_name = model_name

        self.tokenizer = EsmSequenceTokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "esm3_sm_open_v1":
            self.model = ESM3.from_pretrained(model_name).to(self.device)
        else:
            self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.del_special_tokens = del_special_tokens

    def forward(self, seqs: List[str]):
        tokens = torch.tensor([self.tokenizer.encode(seq) for seq in seqs]).to(
            self.device
        )
        embeds = self.model(sequence_tokens=tokens).embeddings
        if self.del_special_tokens:
            embeds = embeds[:, 1:-1]
        return embeds
