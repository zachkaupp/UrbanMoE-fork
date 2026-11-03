import os
import torch
from configs.args_parser import TASKS, create_args
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


def load_embedding(args, name, df, train_df):
    """The embedding table for a modality, standardized with the training split's mean and std."""
    path = os.path.join(args.embedding_dir, f"{args.dataset}_{name}.csv")
    table = pd.read_csv(path, index_col="name")
    if name == "contextual_count":
        table = np.log1p(table)

    names = lambda frame: frame["filename"].str.replace(".tif.tif", ".csv", regex=False)
    missing = sorted(set(names(df)) - set(table.index))
    if missing:
        raise ValueError(f"{path} has no embedding for {len(missing)} tiles, e.g. {missing[:5]}")

    train_table = table.loc[names(train_df)]
    mean, std = train_table.mean(), train_table.std(ddof=0)
    std = std.where(std > 1e-8, 1.0)  # constant columns in the training split
    return ((table - mean) / std).astype(np.float32)


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

    label_columns = [TASKS[task][0] for task in args.tasks]
    mean = [np.mean(train_df[column]) for column in label_columns]
    std = [np.std(train_df[column]) for column in label_columns]

    embedding_modalities = [m for m in args.modalities if m not in ("image", "text", "region")]
    if embedding_modalities and args.dataset != "NewYork":
        raise ValueError(f"{embedding_modalities} embeddings only exist for NewYork")
    embeddings = {name: load_embedding(args, name, df, train_df) for name in embedding_modalities}
    args.embedding_dims = {name: table.shape[1] for name, table in embeddings.items()}

    dataset_args = (args.dataset,)
    dataset_kwargs = dict(transform=transform, tokenizer=tokenizer, is_test=False, mean=mean, std=std,
                          poi_path=poi_path, modalities=args.modalities, embeddings=embeddings,
                          label_columns=label_columns)
    train_set = getDataset(*dataset_args, df_data=train_df, **dataset_kwargs)
    val_set = getDataset(*dataset_args, df_data=val_df, **dataset_kwargs)
    test_set = getDataset(*dataset_args, df_data=test_df, **dataset_kwargs)

    return train_set, val_set, test_set, mean, std


def encode_clip_features(model, datasets, args):
    """Run the frozen CLIP encoders once over every tile, so training doesn't re-encode them each epoch."""
    model.clip.eval()
    for dataset in datasets:
        raw = {"image": dataset.img_tensors, "text": dataset.text_tensors}
        for name in ("image", "text"):
            if name not in dataset.modalities:
                continue
            features = []
            for start in range(0, len(raw[name]), args.batch_size):
                batch = torch.stack(raw[name][start:start + args.batch_size]).to(args.device)
                features.append(model.encode_clip(name, batch).cpu())
            dataset.use_features(name, torch.cat(features))


def main():
    args = create_args()
    set_random_seed(args.seed)
    logger = create_logger(args.logging_dir, args.seed)
    writer = create_writer(os.path.join(args.logging_dir, "tb", str(args.seed)))

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
    encode_clip_features(model, (train_set, val_set, test_set), args)

    data = {
        "train_loader": DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True),
        "val_loader": DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        "test_loader": DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
        "mean": mean,
        "std": std
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val = float("-inf")

    for epoch in range(args.epoch_num):
        train_one_epoch(model, optimizer, data, epoch, args, logger, writer)
        metrics = evaluate(model, data, args, logger, split="val", writer=writer, epoch=epoch)
        # NOTE: best checkpoint uses the mean val R2 over the tasks. Upstream used Carbon val MSE only,
        # which a few extreme carbon tiles dominate. The paper doesn't state its selection criterion
        val_score = np.mean([metrics[f"r2_{TASKS[task][1]}"] for task in args.tasks])
        if val_score > best_val:
            save_checkpoint(model, optimizer, epoch, os.path.join(args.checkpoint_dir, "best.pt"), best=True)
            best_val = val_score

    logger.info("Training finished")
    load_checkpoint(model, os.path.join(args.checkpoint_dir, "best.pt"), args.device)
    evaluate(model, data, args, logger, split="test")


if __name__ == "__main__":
    main()