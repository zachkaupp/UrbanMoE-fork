import torch
import torch.nn as nn
import torch.nn.functional as F
from .experts import Expert
from .loss_utils import contrastive_loss


class UrbanMoE_Regression(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.clip = clip_model
        self.args = args
        self.num_tasks = args.num_tasks

        self.region_embedding = nn.Embedding(
            num_embeddings=args.num_regions,
            embedding_dim=args.region_embedding_dim,
            device=args.device
        )
        nn.init.xavier_normal_(self.region_embedding.weight)

        self.num_task_pairs = args.num_tasks * (args.num_tasks - 1) // 2

        self.image_experts = nn.ModuleDict({
            'all_shared': nn.ModuleList([
                Expert(args.img_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_all_shared_experts)
            ]),
            'pair_shared': nn.ModuleList([
                Expert(args.img_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_pair_shared_experts * self.num_task_pairs)
            ]),
            'task_specific': nn.ModuleList([
                Expert(args.img_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_task_experts * args.num_tasks)
            ])
        })

        self.text_experts = nn.ModuleDict({
            'all_shared': nn.ModuleList([
                Expert(args.text_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_all_shared_experts)
            ]),
            'pair_shared': nn.ModuleList([
                Expert(args.text_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_pair_shared_experts * self.num_task_pairs)
            ]),
            'task_specific': nn.ModuleList([
                Expert(args.text_embedding_dim + args.region_embedding_dim + (
                    args.output_feature_dim if args.use_poi else 0),
                    args.output_feature_dim)
                for _ in range(args.num_task_experts * args.num_tasks)
            ])
        })

        self.gate_kernels_image = nn.ParameterList([
            nn.Parameter(torch.empty(
                args.img_embedding_dim,
                args.num_all_shared_experts +
                args.num_pair_shared_experts * (args.num_tasks - 1) +
                args.num_task_experts,
                device=args.device
            )) for _ in range(args.num_tasks)
        ])
        self.gate_kernels_text = nn.ParameterList([
            nn.Parameter(torch.empty(
                args.text_embedding_dim,
                args.num_all_shared_experts +
                args.num_pair_shared_experts * (args.num_tasks - 1) +
                args.num_task_experts,
                device=args.device
            )) for _ in range(args.num_tasks)
        ])

        self.gate_bias_image = nn.ParameterList([
            nn.Parameter(torch.zeros(
                args.num_all_shared_experts +
                args.num_pair_shared_experts * (args.num_tasks - 1) +
                args.num_task_experts,
                device=args.device
            )) for _ in range(args.num_tasks)
        ])
        self.gate_bias_text = nn.ParameterList([
            nn.Parameter(torch.zeros(
                args.num_all_shared_experts +
                args.num_pair_shared_experts * (args.num_tasks - 1) +
                args.num_task_experts,
                device=args.device
            )) for _ in range(args.num_tasks)
        ])

        for kernel in self.gate_kernels_image + self.gate_kernels_text:
            nn.init.xavier_normal_(kernel)
        for bias in self.gate_bias_image + self.gate_bias_text:
            nn.init.uniform_(bias, -0.1, 0.1)

        self.poi_projection = nn.Linear(args.poi_dim, args.output_feature_dim) if args.use_poi else None

        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.output_feature_dim * 2, args.hidden_num * 2),
                nn.ReLU(),
                nn.Dropout(args.drop_out),
                nn.Linear(args.hidden_num * 2, 1)
            ) for _ in range(args.num_tasks)
        ])

    def sparse_softmax(self, logits, mode="topk", k=2, threshold=0.02):
        probs = F.softmax(logits, dim=-1)
        if mode == "topk":
            k = min(k, logits.size(-1))
            topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, 1.0)
            probs *= mask
        elif mode == "threshold":
            mask = (probs >= threshold).float()
            probs *= mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
        return probs

    def forward(self, args, images, texts, region_ids=None, targets=None, poi_features=None):
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            if texts.dim() == 3 and texts.shape[1] == 1:
                texts = texts.squeeze(1)
            text_features = self.clip.encode_text(texts)

        if poi_features is not None and self.poi_projection is not None:
            poi_features = poi_features.to(image_features.device)
            poi_emb = self.poi_projection(poi_features)
            image_features = torch.cat([image_features, poi_emb], dim=1)
            text_features = torch.cat([text_features, poi_emb], dim=1)

        region_embeds = self.region_embedding(region_ids) if region_ids is not None else \
            torch.zeros(len(image_features), self.region_embedding.embedding_dim, device=image_features.device)
        fused_image = torch.cat([image_features, region_embeds], dim=1)
        fused_text = torch.cat([text_features, region_embeds], dim=1)

        outputs_img, outputs_txt = [], []
        task_pairs = [(i, j) for i in range(self.num_tasks) for j in range(i + 1, self.num_tasks)]

        for task_idx in range(self.num_tasks):
            gate_img = torch.einsum("bi,ie->be", image_features, self.gate_kernels_image[task_idx]) + self.gate_bias_image[task_idx]
            gate_img = self.sparse_softmax(gate_img, mode="threshold", threshold=0)
            gate_txt = torch.einsum("bi,ie->be", text_features, self.gate_kernels_text[task_idx]) + self.gate_bias_text[task_idx]
            gate_txt = self.sparse_softmax(gate_txt, mode="threshold", threshold=0)

            weighted_img = self._aggregate_experts(task_idx, gate_img, fused_image, self.image_experts, task_pairs)
            weighted_txt = self._aggregate_experts(task_idx, gate_txt, fused_text, self.text_experts, task_pairs)
            outputs_img.append(weighted_img)
            outputs_txt.append(weighted_txt)

        outputs = []
        for i in range(self.num_tasks):
            fused = torch.cat([outputs_img[i], outputs_txt[i]], dim=1)
            outputs.append(self.output_layer[i](fused))

        if self.training and targets is not None:
            reg_loss = sum([F.mse_loss(outputs[i], targets[:, i].unsqueeze(1)) for i in range(self.num_tasks)])
            contr_loss = sum([contrastive_loss(outputs_img[i], outputs_txt[i]) for i in range(self.num_tasks)])
            total_loss = reg_loss + 0 * contr_loss
            return torch.cat(outputs, dim=1), total_loss

        return torch.cat(outputs, dim=1)

    def _aggregate_experts(self, task_idx, gate, fused_feat, expert_dict, task_pairs):
        start = 0
        weighted_all, weighted_pair, weighted_task = 0, 0, 0
        num_all = self.args.num_all_shared_experts
        num_pair = self.args.num_pair_shared_experts
        num_task = self.args.num_task_experts

        gate_all = gate[:, start:start + num_all]; start += num_all
        if num_all > 0:
            all_out = torch.stack([expert(fused_feat) for expert in expert_dict['all_shared']], dim=2)
            weighted_all = torch.sum(all_out * gate_all.unsqueeze(1), dim=2)

        related_pairs = [i for i, (a, b) in enumerate(task_pairs) if task_idx in (a, b)]
        if num_pair > 0:
            gate_pairs = gate[:, start:start + len(related_pairs) * num_pair]; start += len(related_pairs) * num_pair
            for i, pid in enumerate(related_pairs):
                start_i, end_i = i * num_pair, (i + 1) * num_pair
                experts = expert_dict['pair_shared'][pid * num_pair:(pid + 1) * num_pair]
                pair_out = torch.stack([e(fused_feat) for e in experts], dim=2)
                weighted_pair += torch.sum(pair_out * gate_pairs[:, start_i:end_i].unsqueeze(1), dim=2)

        gate_task = gate[:, start:start + num_task]
        experts = expert_dict['task_specific'][task_idx * num_task:(task_idx + 1) * num_task]
        task_out = torch.stack([e(fused_feat) for e in experts], dim=2)
        weighted_task = torch.sum(task_out * gate_task.unsqueeze(1), dim=2)

        return weighted_all + weighted_pair + weighted_task