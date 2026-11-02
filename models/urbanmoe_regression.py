import torch
import torch.nn as nn
import torch.nn.functional as F
from .experts import Expert


class UrbanMoE_Regression(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.clip = clip_model
        self.args = args
        self.num_tasks = args.num_tasks
        self.branches = list(args.modalities)

        self.region_embedding = nn.Embedding(
            num_embeddings=args.num_regions,
            embedding_dim=args.region_embedding_dim,
            device=args.device
        )
        nn.init.xavier_normal_(self.region_embedding.weight)

        self.num_task_pairs = args.num_tasks * (args.num_tasks - 1) // 2

        # one set of experts and gates per branch (modality), each with its own input size
        branch_dims = {name: self._input_dim(name) for name in self.branches}
        self.experts = nn.ModuleDict({
            name: nn.ModuleDict({
                'all_shared': nn.ModuleList([
                    Expert(dim, args.output_feature_dim)
                    for _ in range(args.num_all_shared_experts)
                ]),
                'pair_shared': nn.ModuleList([
                    Expert(dim, args.output_feature_dim)
                    for _ in range(args.num_pair_shared_experts * self.num_task_pairs)
                ]),
                'task_specific': nn.ModuleList([
                    Expert(dim, args.output_feature_dim)
                    for _ in range(args.num_task_experts * args.num_tasks)
                ])
            }) for name, dim in branch_dims.items()
        })

        num_gate_outputs = (
            args.num_all_shared_experts +
            args.num_pair_shared_experts * (args.num_tasks - 1) +
            args.num_task_experts
        )
        self.gate_kernels = nn.ModuleDict({
            name: nn.ParameterList([
                nn.Parameter(torch.empty(dim, num_gate_outputs, device=args.device))
                for _ in range(args.num_tasks)
            ]) for name, dim in branch_dims.items()
        })
        self.gate_biases = nn.ModuleDict({
            name: nn.ParameterList([
                nn.Parameter(torch.zeros(num_gate_outputs, device=args.device))
                for _ in range(args.num_tasks)
            ]) for name in branch_dims
        })

        for name in self.branches:
            for kernel in self.gate_kernels[name]:
                nn.init.xavier_normal_(kernel)
        for name in self.branches:
            for bias in self.gate_biases[name]:
                nn.init.uniform_(bias, -0.1, 0.1)

        # TODO: paper uses d_p=15 (POI embedding) and d_r=6; here POI counts are projected to
        # output_feature_dim (32). Unclear whether the paper applies a learned projection to the counts
        self.poi_projection = nn.Linear(args.poi_dim, args.output_feature_dim) if args.use_poi else None

        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.output_feature_dim * len(self.branches), args.hidden_num * 2),
                nn.ReLU(),
                nn.Dropout(args.drop_out),
                nn.Linear(args.hidden_num * 2, 1)
            ) for _ in range(args.num_tasks)
        ])

    def _input_dim(self, name):
        """Size of a branch's expert input: the modality vector, the region embedding and the POI projection."""
        args = self.args
        modality_dim = {
            'image': args.img_embedding_dim,
            'text': args.text_embedding_dim,
        }.get(name) or args.embedding_dims[name]
        return modality_dim + args.region_embedding_dim + (args.output_feature_dim if args.use_poi else 0)

    def sparse_softmax(self, logits, mode="topk", k=2, threshold=0.02):
        probs = F.softmax(logits, dim=-1)
        if mode == "topk":
            k = min(k, logits.size(-1))
            topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, 1.0)
            probs = probs * mask
        elif mode == "threshold":
            mask = (probs >= threshold).float()
            probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
        return probs

    def forward(self, args, inputs, region_ids=None, targets=None, poi_features=None):
        features = {}
        with torch.no_grad():
            if 'image' in self.branches:
                features['image'] = self.clip.encode_image(inputs['image'])
            if 'text' in self.branches:
                texts = inputs['text']
                if texts.dim() == 3 and texts.shape[1] == 1:
                    texts = texts.squeeze(1)
                features['text'] = self.clip.encode_text(texts)
        for name in self.branches:
            if name not in features:
                features[name] = inputs[name]
        reference = features[self.branches[0]]

        if poi_features is not None and self.poi_projection is not None:
            poi_features = poi_features.to(reference.device)
            poi_emb = self.poi_projection(poi_features)
            features = {name: torch.cat([feature, poi_emb], dim=1) for name, feature in features.items()}

        region_embeds = self.region_embedding(region_ids) if region_ids is not None else \
            torch.zeros(len(reference), self.region_embedding.embedding_dim, device=reference.device)
        fused = {name: torch.cat([feature, region_embeds], dim=1) for name, feature in features.items()}

        branch_outputs = {name: [] for name in self.branches}
        task_pairs = [(i, j) for i in range(self.num_tasks) for j in range(i + 1, self.num_tasks)]

        # TODO: threshold=0 keeps every expert, so the gating is dense. The paper (Eq. 5, App. C)
        # zeroes gate weights <= eps and finds eps=0.01 best; eps=0 is reported as "still competitive"
        for task_idx in range(self.num_tasks):
            for name in self.branches:
                gate = torch.einsum("bi,ie->be", fused[name], self.gate_kernels[name][task_idx]) + self.gate_biases[name][task_idx]
                gate = self.sparse_softmax(gate, mode="threshold", threshold=0)
                branch_outputs[name].append(
                    self._aggregate_experts(task_idx, gate, fused[name], self.experts[name], task_pairs))

        outputs = []
        for i in range(self.num_tasks):
            combined = torch.cat([branch_outputs[name][i] for name in self.branches], dim=1)
            outputs.append(self.output_layer[i](combined))

        if self.training and targets is not None:
            reg_loss = sum([F.mse_loss(outputs[i], targets[:, i].unsqueeze(1)) for i in range(self.num_tasks)])
            # NOTE: intentional, matches the paper (Eq. 8): the objective is weighted MSE only. The contrastive
            # term between the image and text outputs had weight 0, so it is no longer computed
            return torch.cat(outputs, dim=1), reg_loss

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
