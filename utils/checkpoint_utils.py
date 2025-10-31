import os
import torch


def save_checkpoint(model, optimizer, epoch, path, best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)
    if best:
        print(f"Saved best model to {path}")
    else:
        print(f"Saved checkpoint to {path}")


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from {path}")
    return ckpt