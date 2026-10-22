import argparse
import torch


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Shanghai",
                        choices=["Beijing", "Shanghai", "NewYork"])


    parser.add_argument("--train_dataset_ratio", type=float, default=0.6,
                        help="train_dataset_ratio")
    parser.add_argument("--val_dataset_ratio", type=float, default=0.2,
                        help="val_dataset_ratio")

    parser.add_argument("--img_embedding_dim", type=int, default=768)
    parser.add_argument("--text_embedding_dim", type=int, default=768)
    parser.add_argument("--region_embedding_dim", type=int, default=6)
    parser.add_argument("--num_regions", type=int, default=1848)
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--output_feature_dim", type=int, default=32)
    parser.add_argument("--hidden_num", type=int, default=8)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu"])
    parser.add_argument("--drop_out", type=float, default=0.01)


    parser.add_argument("--num_all_shared_experts", type=int, default=4)
    parser.add_argument("--num_pair_shared_experts", type=int, default=2)
    parser.add_argument("--num_task_experts", type=int, default=8)


    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=1)

    parser.add_argument("--seed", type=int, default=132)
    parser.add_argument("--device", type=str,
                        default=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument("--logging_dir", type=str,
                        default="logs/")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoint/")


    parser.add_argument("--indicator1", type=str, default="c_value")
    parser.add_argument("--indicator2", type=str, default="pop_value")
    parser.add_argument("--indicator3", type=str, default="light_value")


    parser.add_argument("--use_poi", type=bool, default=True)
    parser.add_argument("--poi_dim", type=int, default=0)

    return parser.parse_args()