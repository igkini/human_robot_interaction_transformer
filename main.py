import torch
from torch.utils.data import DataLoader, random_split

from model.model import HumanRobotInteractionTransformer
from model.model_params import ModelParams
from dataset.torch_dataset import TrajectoryPredictionDataset

from utils.training import train

TRAIN_PATH = "data/oxford/train_val"
VAL_PATH     = "data/oxford/test"
CKPT_DIR      = "checkpoints"

seq_len  = 10
pred_len = 10

train_dataset = TrajectoryPredictionDataset(
    data_path=TRAIN_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

test_dataset = TrajectoryPredictionDataset(
    data_path=VAL_PATH,
    seq_len=seq_len,
    pred_len=pred_len,
)

test_ratio = 0.4
test_size  = max(1, int(len(test_dataset) * test_ratio))
val_size = max(1, len(test_dataset) - test_size)
val_dataset, test_dataset = random_split(
    test_dataset, [val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 16
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = ModelParams()
model  = HumanRobotInteractionTransformer(params=params).to(device)

history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=20,          
    patience=8,    
    learning_rate=1e-4,
    weight_decay=1e-5,
    ckpt_dir=CKPT_DIR,
)

print("Done. Last epoch losses:", {k: v[-1] for k, v in history.items() if len(v) > 0})
