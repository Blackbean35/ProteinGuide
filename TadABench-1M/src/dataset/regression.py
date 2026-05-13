import os
from typing import Tuple
from torch.utils.data import Dataset
from datasets import load_from_disk


CODON2AA = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

AA2CODON = {}
for k, v in CODON2AA.items():
    if v not in AA2CODON:
        AA2CODON[v] = []
    AA2CODON[v].append(k)


def DNA2AA(DNA: str):
    assert len(DNA) % 3 == 0, f"Invalid DNA length: {len(DNA)}"
    return "".join(CODON2AA[DNA[i : i + 3]] for i in range(0, len(DNA), 3))


def DNA2RNA(DNA: str):
    return DNA.replace("T", "U")


def modality_map(seq_type: str, seq: str):
    if seq_type == "AA":
        seq = DNA2AA(seq)
    if seq_type == "RNA":
        seq = DNA2RNA(seq)
    return seq


class RegressionDataset(Dataset):
    def __init__(self, cfg, split: str = ""):
        self.cfg = cfg
        self.seq_type = cfg.seq_type
        self.split = split
        # Optional legacy: cfg.huggingface_dataset (unused in offline-only mode)
        self.dataset_name = getattr(cfg, "huggingface_dataset", None)
        self.local_dataset_dir = getattr(
            cfg, "local_dataset_dir", os.getenv("NB1M_LOCAL_DATASET_DIR", "data")
        )
        self.length = cfg.length
        self.return_seq = getattr(cfg, "return_seq", False)
        self.normalize_label = getattr(cfg, "normalize_label", False)
        # optional max samples for quick runs
        self.max_samples = getattr(cfg, "max_samples", None)
        self.max_train_samples = getattr(cfg, "max_train_samples", None)
        self.max_val_samples = getattr(cfg, "max_val_samples", None)
        self.max_test_samples = getattr(cfg, "max_test_samples", None)

        self.load_data()

    def load_data(self):
        # Prefer local offline dataset if present; else fall back to HF Dataset ID
        local_path = os.path.join(
            self.local_dataset_dir, f"all.{self.seq_type}.{self.split}"
        )

        if os.path.isdir(local_path):
            data = load_from_disk(local_path)
        else:
            raise FileNotFoundError(
                f"Local dataset not found at '{local_path}'. "
                "This repo is configured for offline review; please include the data/ directory."
            )
        self.data = data["Sequence"]
        self.labels = data["Value"]

        # Optional: cap samples for quick/smoke runs
        if self.split == "train":
            max_n = self.max_train_samples
        elif self.split == "val":
            max_n = self.max_val_samples
        else:
            max_n = self.max_test_samples
        # fallback to global max_samples
        if max_n is None:
            max_n = self.max_samples
        if max_n is not None and max_n > 0:
            self.data = self.data[:max_n]
            self.labels = self.labels[:max_n]

        if self.normalize_label:
            self.max_label = max(self.labels)
            self.min_label = min(self.labels)
            self.labels = [
                (label - self.min_label) / (self.max_label - self.min_label)
                for label in self.labels
            ]
            print(
                f"Normalizing labels from [{self.min_label}, {self.max_label}] to [{min(self.labels)}, {max(self.labels)}]"
            )

        self.num_samples = len(self.data)
        self.seqs = set(self.data)
        print(f"Loaded {len(self.data)} sequences ({len(self.seqs)} unique)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[Tuple[str], Tuple[str]]:
        data, label = self.data[idx], self.labels[idx]

        if self.return_seq:
            return (data, self.data[idx]), label
        else:
            return data, label
