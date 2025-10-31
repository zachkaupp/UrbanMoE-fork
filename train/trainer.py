import torch
import time
from tqdm import tqdm
from .metrics import compute_multitask_metrics


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, optimizer, data, epoch, args, logger, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    dataloader = data["train_loader"]
    mse_meter = AverageMeter()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images, texts, region_id, y, poi = batch
        images, texts, region_id, y, poi = images.to(device), texts.to(device), region_id.to(device), y.to(device), poi.to(device)
        optimizer.zero_grad()
        preds, loss = model(args, images, texts, region_id, y, poi)
        loss.backward()
        optimizer.step()
        mse_meter.update(loss.item(), len(images))

        if batch_idx % args.log_every_n_steps == 0:
            logger.info(f"[Epoch {epoch}] Step {batch_idx} | Loss: {loss.item():.5f}")
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), epoch * len(dataloader) + batch_idx)

    logger.info(f"Epoch {epoch} average loss: {mse_meter.avg:.5f}")


def evaluate(model, data, args, logger, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataloader = data[f"{split}_loader"]
    all_y, all_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            images, texts, region_id, y, poi = batch
            images, texts, region_id, y, poi = images.to(device), texts.to(device), region_id.to(device), y.to(device), poi.to(device)
            preds = model(args, images, texts, region_id, y, poi)
            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    import numpy as np
    all_y, all_pred = np.concatenate(all_y), np.concatenate(all_pred)
    metrics = compute_multitask_metrics(all_y, all_pred, data["mean"], data["std"])
    logger.info(f"{split.capitalize()} metrics: {metrics}")
    return metrics