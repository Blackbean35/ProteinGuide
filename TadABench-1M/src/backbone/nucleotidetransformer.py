import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM


class NucleotideTransformerModel(nn.Module):
    model_names = [
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "InstaDeepAI/nucleotide-transformer-500m-1000g",
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "InstaDeepAI/agro-nucleotide-transformer-1b",
    ]

    def __init__(self, model_name: str, del_special_tokens: bool = True):
        assert model_name in self.model_names, (
            f"Model name {model_name} not found in {self.model_names}"
        )
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # {'<unk>': 0, '<pad>': 1, '<mask>': 2, '<cls>': 3, '<eos>': 4, '<bos>': 5, 'AAAAAA': 6, ...}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.del_special_tokens = del_special_tokens

    def forward(self, seqs: List[str]):
        inputs = self.tokenizer(seqs, return_tensors="pt").to(self.device)
        outputs = self.model(output_hidden_states=True, **inputs)
        embeddings = outputs.hidden_states[-1]
        if self.del_special_tokens:
            embeddings = embeddings[:, 1:]
        return embeddings
