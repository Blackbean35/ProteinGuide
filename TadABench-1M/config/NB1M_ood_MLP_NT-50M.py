task = "run"


# Dataset
dataset_type = "RegressionDataset"
train_val_test = True
# Offline review: only reads from local data/
del_special_tokens = True
local_dataset_dir = "data"  # prefer local offline dataset


# Embed from files
use_embed_mapper = False
mem_cache = False
frozen_backbone = True
dtype = {"head": "fp32", "backbone": "bf16"}  # You can change the dtype of the backbone
embed_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
seq_type = "DNA"
length = 501


# Evaluation and saving
eval_before_train = True
eval_interval = 1
save_interval = -1
save_dir = "ckpt/linear_ood"


# Optimizer and learning rate
num_epochs = 20
batch_size = 256
optimizer_type = "AdamW"
learning_rate = [{"head": 3e-5}, {"head": 1e-4}, {"head": 3e-4}]
scheduler_kwargs = {
    "type": "CosineAnnealingLR",
    "is_epoch": True,
    "num_warmup_steps": 1,
    "num_training_steps": num_epochs,
}
weight_decay = {"head": 1e-4}
test_batch_size = 256
loss_type = ["mse"]
evaluation = [
    [
        "sp",
        "recall_at_10pct",
        "ndcg_at_10pct",
    ]
]


# Model
head_model_type = "MLP"
num_tokens = 86 + 2 * (0 if del_special_tokens else 1)
embed_dim = 512
hidden_sizes = [
    [int(501 * 4096 * 16 / num_tokens / embed_dim)],
]
num_layers = 2
activation = "ReLU"


# Wandb config
use_wandb = False  # False if you don't want to use wandb
tag = "MLP_ESM2-35M"
wandb_proj_name = None  # None if you don't want to use wandb
entity = None  # None if you don't want to use wandb
