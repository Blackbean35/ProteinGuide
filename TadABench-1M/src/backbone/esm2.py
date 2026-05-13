from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List
import torch
import torch.nn as nn


class ESM2Model(nn.Module):
    model_names = [
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t48_15B_UR50D",
        "facebook/esm2_t36_3B_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t6_8M_UR50D",
    ]

    def __init__(
        self, model_name: str, pretrained: bool, del_special_tokens: bool
    ) -> None:
        assert model_name in self.model_names, (
            f"Model {model_name} not found in {self.model_names}."
        )
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pretrained:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True).to(
                self.device
            )
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.del_special_tokens = del_special_tokens

    def forward(self, seqs: List[str]):
        inputs = self.tokenizer(seqs, return_tensors="pt").to(self.device)
        output = self.model(**inputs)
        embeddings = output.last_hidden_state
        if self.del_special_tokens:
            embeddings = embeddings[:, 1:-1]
        return embeddings
