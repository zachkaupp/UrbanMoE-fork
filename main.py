import os
import torch
from configs.args_parser import create_args
from data.data_utils import set_random_seed, count_all_parameters, count_trainable_parameters
from data.dataset_clip import getDataset
from models.urbanmoe_regression import UrbanMoE_Regression
from train.trainer import train_one_epoch, evaluate
from utils.logger_utils import create_logger, create_writer
from utils.checkpoint_utils import save_checkpoint, load_checkpoint

import open_clip
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def create_datasets(args, transform, tokenizer):
    if args.dataset == "Shanghai":
        df = pd.read_csv("data/shanghai_merged_normalized.csv")
        poi_path = "data/shanghai_poi_category_counts.csv"
    elif args.dataset == "Beijing":
        df = pd.read_csv("data/Beijing_merged_normalized.csv")
        poi_path = None
    elif args.dataset == "NewYork":
        df = pd.read_csv("data/NewYork_merged_normalized.csv")
        poi_path = "data/NY_merged_top_category_counts.csv"
    else:
        raise ValueError("Unsupported dataset")

    df = df.sample(frac=1).reset_index(drop=True)
    train_end = int(len(df) * args.train_dataset_ratio)
    val_end = int(len(df) * (1-args.val_dataset_ratio))

    train_df, val_df, test_df = df[:train_end], df[train_end:val_end], df[val_end:]

    mean = [np.mean(train_df[args.indicator1]), np.mean(train_df[args.indicator2]), np.mean(train_df[args.indicator3])]
    std = [np.std(train_df[args.indicator1]), np.std(train_df[args.indicator2]), np.std(train_df[args.indicator3])]

    train_set = getDataset(args.dataset, train_df, transform, tokenizer, False, mean, std, poi_path)
    val_set = getDataset(args.dataset, val_df, transform, tokenizer, False, mean, std, poi_path)
    test_set = getDataset(args.dataset, test_df, transform, tokenizer, False, mean, std, poi_path)

    return train_set, val_set, test_set, mean, std


def main():
    args = create_args()
    set_random_seed(args.seed)
    logger = create_logger(args.logging_dir, args.seed)
    writer = create_writer(os.path.join(args.logging_dir, "tb"))

    model_name = "ViT-L-14"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    ckpt = torch.load("RemoteCLIP-ViT-L-14.pt", map_location="cpu")
    clip_model.load_state_dict(ckpt)

    for p in clip_model.parameters():
        p.requires_grad = False

    train_set, val_set, test_set, mean, std = create_datasets(args, preprocess, tokenizer)
    args.poi_dim = len(train_set.poi_columns) if hasattr(train_set, "poi_columns") else 0

    model = UrbanMoE_Regression(clip_model, args).to(args.device)
    logger.info(f"Total params: {count_all_parameters(model)} | Trainable: {count_trainable_parameters(model)}")

    data = {
        "train_loader": DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True),
        "val_loader": DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        "test_loader": DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
        "mean": mean,
        "std": std
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val = float("inf")

    for epoch in range(args.epoch_num):
        train_one_epoch(model, optimizer, data, epoch, args, logger, writer)
        metrics = evaluate(model, data, args, logger, split="val")
        if metrics["mse_Carbon"] < best_val:
            save_checkpoint(model, optimizer, epoch, os.path.join(args.checkpoint_dir, "best.pt"), best=True)
            best_val = metrics["mse_Carbon"]

    logger.info("Training finished")
    load_checkpoint(model, os.path.join(args.checkpoint_dir, "best.pt"), args.device)
    evaluate(model, data, args, logger, split="test")


if __name__ == "__main__":
    main()