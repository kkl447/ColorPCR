from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform


def _empty_diagnostics(device):
    zero = torch.zeros((), device=device)
    minus_one = zero.new_tensor(-1.0)
    return {
        'mac_num_candidates': zero,
        'mac_num_edges': zero,
        'mac_num_cliques': zero,
        'mac_num_selected_cliques': zero,
        'mac_best_clique_size': zero,
        'mac_best_score': minus_one,
        'mac_fallback': zero.new_tensor(1.0),
        'mac_fallback_reason': 'not_run',
    }


def _safe_normalize(x, dim=-1, eps=1e-8):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


class MACRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        voxel_size: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        patch_topk: int = 16,
        max_corr: int = 1000,
        use_sog: bool = True,
        edge_threshold: float = 0.99,
        dist_sigma_factor: float = 10.0,
        min_clique_size: int = 3,
        normal_sigma: float = 0.10,
        use_normal_fog: bool = True,
        num_refinement_steps: int = 5,
        max_hypotheses: Optional[int] = None,
        use_weighted_svd: bool = False,
        use_weighted_scoring: bool = False,
    ):
        super(MACRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.voxel_size = voxel_size
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.patch_topk = patch_topk
        self.max_corr = max_corr
        self.use_sog = use_sog
        self.edge_threshold = edge_threshold
        self.dist_sigma_factor = dist_sigma_factor
        self.min_clique_size = min_clique_size
        self.normal_sigma = normal_sigma
        self.use_normal_fog = use_normal_fog
        self.num_refinement_steps = num_refinement_steps
        self.max_hypotheses = max_hypotheses
        self.use_weighted_svd = use_weighted_svd
        self.use_weighted_scoring = use_weighted_scoring

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))
        batch_size, ref_length, src_length = score_mat.shape
        device = score_mat.device

        ref_topk = min(int(self.k), src_length)
        src_topk = min(int(self.k), ref_length)
        batch_indices = torch.arange(batch_size, device=device)

        ref_topk_scores, ref_topk_indices = score_mat.topk(k=ref_topk, dim=2)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, ref_topk)
        ref_indices = torch.arange(ref_length, device=device).view(1, ref_length, 1).expand(batch_size, -1, ref_topk)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        src_topk_scores, src_topk_indices = score_mat.topk(k=src_topk, dim=1)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, src_topk, src_length)
        src_indices = torch.arange(src_length, device=device).view(1, 1, src_length).expand(batch_size, src_topk, -1)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat) if self.mutual else torch.logical_or(ref_corr_mat, src_corr_mat)
        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
        return torch.logical_and(corr_mat, mask_mat)

    def _extract_correspondences(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
        ref_knn_normals=None,
        src_knn_normals=None,
    ):
        score_mat = torch.exp(score_mat)
        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)
        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        src_corr_points = src_knn_points[batch_indices, src_indices]
        corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        ref_corr_normals = None
        src_corr_normals = None
        if ref_knn_normals is not None and src_knn_normals is not None:
            ref_corr_normals = ref_knn_normals[batch_indices, ref_indices]
            src_corr_normals = src_knn_normals[batch_indices, src_indices]

        return (
            ref_corr_points,
            src_corr_points,
            corr_scores,
            batch_indices,
            ref_corr_normals,
            src_corr_normals,
        )

    def _select_candidates(self, corr_scores, patch_indices):
        if corr_scores.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=corr_scores.device)

        selected = []
        unique_patches = torch.unique(patch_indices)
        for patch_index in unique_patches:
            masks = patch_indices == patch_index
            local_indices = torch.nonzero(masks, as_tuple=True)[0]
            k = min(int(self.patch_topk), local_indices.numel())
            if k <= 0:
                continue
            _, order = corr_scores[local_indices].topk(k=k, largest=True)
            selected.append(local_indices[order])
        if len(selected) == 0:
            return torch.empty(0, dtype=torch.long, device=corr_scores.device)

        candidate_indices = torch.cat(selected, dim=0)
        if self.max_corr is not None and self.max_corr > 0 and candidate_indices.numel() > self.max_corr:
            _, order = corr_scores[candidate_indices].topk(k=int(self.max_corr), largest=True)
            candidate_indices = candidate_indices[order]
        return candidate_indices

    def _build_graph(self, ref_points, src_points, ref_normals=None, src_normals=None):
        num_corr = ref_points.shape[0]
        if num_corr < self.min_clique_size:
            return None, 0

        src_dists = torch.cdist(src_points.unsqueeze(0), src_points.unsqueeze(0)).squeeze(0)
        ref_dists = torch.cdist(ref_points.unsqueeze(0), ref_points.unsqueeze(0)).squeeze(0)
        diff = (src_dists - ref_dists).abs()
        sigma_d = max(float(self.dist_sigma_factor) * float(self.voxel_size), 1e-6)
        weights = torch.exp(-(diff ** 2) / (2.0 * sigma_d * sigma_d))

        if self.use_normal_fog and ref_normals is not None and src_normals is not None:
            ref_normals = _safe_normalize(ref_normals)
            src_normals = _safe_normalize(src_normals)
            ref_cos = torch.matmul(ref_normals, ref_normals.transpose(0, 1)).clamp(-1.0, 1.0)
            src_cos = torch.matmul(src_normals, src_normals.transpose(0, 1)).clamp(-1.0, 1.0)
            ref_sin = torch.sqrt((1.0 - ref_cos ** 2).clamp(min=0.0))
            src_sin = torch.sqrt((1.0 - src_cos ** 2).clamp(min=0.0))
            normal_diff = (src_sin - ref_sin).abs()
            sigma_n = max(float(self.normal_sigma), 1e-6)
            weights = weights * torch.exp(-(normal_diff ** 2) / (2.0 * sigma_n * sigma_n))

        weights.fill_diagonal_(0.0)
        graph_weights = weights * torch.matmul(weights, weights) if self.use_sog else weights
        graph_weights.fill_diagonal_(0.0)
        adjacency = torch.gt(graph_weights, float(self.edge_threshold))
        adjacency.fill_diagonal_(False)
        num_edges = int(adjacency.sum().item() // 2)
        return graph_weights, num_edges

    def _find_maximal_cliques(self, adjacency) -> List[Tuple[int, ...]]:
        num_nodes = adjacency.shape[0]
        adj_cpu = adjacency.detach().cpu()
        adj_bits = []
        for i in range(num_nodes):
            bits = 0
            neighbors = torch.nonzero(adj_cpu[i], as_tuple=True)[0].tolist()
            for j in neighbors:
                bits |= 1 << int(j)
            adj_bits.append(bits)

        all_nodes = (1 << num_nodes) - 1
        cliques = []

        def bits_iter(bits):
            while bits:
                bit = bits & -bits
                yield bit.bit_length() - 1
                bits ^= bit

        def popcount(bits):
            return bin(bits).count('1')

        def is_clique(bits):
            for node in bits_iter(bits):
                others = bits & ~(1 << node)
                if others & ~adj_bits[node]:
                    return False
            return True

        def bron_kerbosch(r, p, x):
            if self.max_hypotheses is not None and len(cliques) >= int(self.max_hypotheses):
                return
            if p == 0 and x == 0:
                if len(r) >= int(self.min_clique_size):
                    cliques.append(tuple(r))
                return
            if p != 0 and is_clique(p):
                clique = r + list(bits_iter(p))
                clique_bits = p
                is_maximal = True
                for node in bits_iter(x):
                    if clique_bits & ~adj_bits[node] == 0:
                        is_maximal = False
                        break
                if is_maximal and len(clique) >= int(self.min_clique_size):
                    cliques.append(tuple(clique))
                return

            union = p | x
            if union:
                pivot = max(bits_iter(union), key=lambda node: popcount(p & adj_bits[node]))
                candidates = p & ~adj_bits[pivot]
            else:
                candidates = p

            for node in list(bits_iter(candidates)):
                bit = 1 << node
                bron_kerbosch(r + [node], p & adj_bits[node], x & adj_bits[node])
                p &= ~bit
                x |= bit
                if self.max_hypotheses is not None and len(cliques) >= int(self.max_hypotheses):
                    return

        bron_kerbosch([], all_nodes, 0)
        return cliques

    def _select_node_guided_cliques(self, cliques, graph_weights):
        if len(cliques) == 0:
            return []

        best_by_node: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        for clique in cliques:
            if len(clique) < self.min_clique_size:
                continue
            indices = torch.as_tensor(clique, dtype=torch.long, device=graph_weights.device)
            sub_weights = graph_weights.index_select(0, indices).index_select(1, indices)
            denom = max(len(clique) * (len(clique) - 1), 1)
            score = float(sub_weights.sum().item() / denom)
            for node in clique:
                old = best_by_node.get(node)
                if old is None or score > old[0]:
                    best_by_node[node] = (score, clique)

        seen = set()
        selected = []
        for _, clique in best_by_node.values():
            key = tuple(sorted(clique))
            if key not in seen:
                selected.append(key)
                seen.add(key)
        return selected

    def _estimate_transform(self, src_points, ref_points, weights=None):
        if src_points.shape[0] < 3:
            return None
        if weights is None:
            weights = torch.ones(src_points.shape[0], device=src_points.device, dtype=src_points.dtype)
        weights = weights.clamp(min=0.0)
        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum <= 0:
            return None
        weights = weights / weight_sum.clamp(min=1e-8)

        src_centroid = (src_points * weights[:, None]).sum(dim=0, keepdim=True)
        ref_centroid = (ref_points * weights[:, None]).sum(dim=0, keepdim=True)
        src_centered = src_points - src_centroid
        ref_centered = ref_points - ref_centroid
        cov = src_centered.transpose(0, 1).matmul(weights[:, None] * ref_centered)
        try:
            cov_cpu = cov.detach().cpu()
            u_cpu, _, v_cpu = torch.svd(cov_cpu)
            eye_cpu = torch.eye(3, device=cov_cpu.device, dtype=cov_cpu.dtype)
            det_cpu = torch.det(v_cpu.matmul(u_cpu.transpose(-2, -1)))
            eye_cpu[-1, -1] = -1.0 if det_cpu.item() < 0.0 else 1.0
            rotation_cpu = v_cpu.matmul(eye_cpu).matmul(u_cpu.transpose(-2, -1))
        except RuntimeError:
            return None
        rotation = rotation_cpu.to(device=src_points.device, dtype=src_points.dtype)
        translation = ref_centroid.squeeze(0) - rotation.matmul(src_centroid.squeeze(0))
        transform = torch.eye(4, device=src_points.device, dtype=src_points.dtype)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        if not torch.isfinite(transform).all():
            return None
        return transform

    def _score_transform(self, transform, ref_points, src_points, corr_scores):
        aligned_src_points = apply_transform(src_points, transform)
        residuals = torch.linalg.norm(ref_points - aligned_src_points, dim=1)
        clipped = residuals.clamp(max=float(self.acceptance_radius))
        if self.use_weighted_scoring:
            weights = corr_scores.clamp(min=0.0)
            if weights.sum() > 0:
                return (clipped * weights).sum() / weights.sum().clamp(min=1e-8)
        return clipped.mean()

    def _refine_transform(self, transform, ref_points, src_points, corr_scores):
        best_transform = transform
        for _ in range(max(int(self.num_refinement_steps), 0)):
            aligned_src_points = apply_transform(src_points, best_transform)
            residuals = torch.linalg.norm(ref_points - aligned_src_points, dim=1)
            masks = residuals < float(self.acceptance_radius)
            if masks.sum().item() < 3:
                break
            weights = corr_scores[masks] if self.use_weighted_svd else None
            refined = self._estimate_transform(src_points[masks], ref_points[masks], weights)
            if refined is None:
                break
            best_transform = refined
        return best_transform

    def _run_mac(self, ref_points, src_points, corr_scores, ref_normals=None, src_normals=None):
        diagnostics = _empty_diagnostics(ref_points.device)
        if ref_points.shape[0] < 3:
            diagnostics['mac_fallback_reason'] = 'too_few_correspondences'
            return None, diagnostics

        graph_weights, num_edges = self._build_graph(ref_points, src_points, ref_normals, src_normals)
        diagnostics['mac_num_candidates'] = ref_points.new_tensor(float(ref_points.shape[0]))
        diagnostics['mac_num_edges'] = ref_points.new_tensor(float(num_edges))
        if graph_weights is None or num_edges == 0:
            diagnostics['mac_fallback_reason'] = 'empty_graph'
            return None, diagnostics

        adjacency = torch.gt(graph_weights, float(self.edge_threshold))
        adjacency.fill_diagonal_(False)
        cliques = self._find_maximal_cliques(adjacency)
        diagnostics['mac_num_cliques'] = ref_points.new_tensor(float(len(cliques)))
        if len(cliques) == 0:
            diagnostics['mac_fallback_reason'] = 'empty_cliques'
            return None, diagnostics

        selected_cliques = self._select_node_guided_cliques(cliques, graph_weights)
        diagnostics['mac_num_selected_cliques'] = ref_points.new_tensor(float(len(selected_cliques)))
        if len(selected_cliques) == 0:
            diagnostics['mac_fallback_reason'] = 'empty_selected_cliques'
            return None, diagnostics

        best_transform = None
        best_score = None
        best_clique_size = 0
        for clique in selected_cliques:
            if len(clique) < self.min_clique_size:
                continue
            indices = torch.as_tensor(clique, dtype=torch.long, device=ref_points.device)
            weights = corr_scores[indices] if self.use_weighted_svd else None
            transform = self._estimate_transform(src_points[indices], ref_points[indices], weights)
            if transform is None:
                continue
            score = self._score_transform(transform, ref_points, src_points, corr_scores)
            if not torch.isfinite(score):
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_transform = transform
                best_clique_size = len(clique)

        if best_transform is None:
            diagnostics['mac_fallback_reason'] = 'no_valid_hypothesis'
            return None, diagnostics

        best_transform = self._refine_transform(best_transform, ref_points, src_points, corr_scores)
        diagnostics['mac_best_clique_size'] = ref_points.new_tensor(float(best_clique_size))
        diagnostics['mac_best_score'] = best_score.detach()
        diagnostics['mac_fallback'] = ref_points.new_tensor(0.0)
        diagnostics['mac_fallback_reason'] = ''
        return best_transform, diagnostics

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
        ref_knn_normals=None,
        src_knn_normals=None,
    ):
        (
            global_ref_corr_points,
            global_src_corr_points,
            global_corr_scores,
            patch_indices,
            global_ref_corr_normals,
            global_src_corr_normals,
        ) = self._extract_correspondences(
            ref_knn_points,
            src_knn_points,
            ref_knn_masks,
            src_knn_masks,
            score_mat,
            global_scores,
            ref_knn_normals=ref_knn_normals,
            src_knn_normals=src_knn_normals,
        )

        candidate_indices = self._select_candidates(global_corr_scores, patch_indices)
        if candidate_indices.numel() == 0:
            diagnostics = _empty_diagnostics(ref_knn_points.device)
            diagnostics['mac_fallback_reason'] = 'empty_candidates'
            return global_ref_corr_points, global_src_corr_points, global_corr_scores, None, diagnostics

        ref_points = global_ref_corr_points[candidate_indices]
        src_points = global_src_corr_points[candidate_indices]
        corr_scores = global_corr_scores[candidate_indices]
        ref_normals = global_ref_corr_normals[candidate_indices] if global_ref_corr_normals is not None else None
        src_normals = global_src_corr_normals[candidate_indices] if global_src_corr_normals is not None else None

        transform, diagnostics = self._run_mac(ref_points, src_points, corr_scores, ref_normals, src_normals)
        return global_ref_corr_points, global_src_corr_points, global_corr_scores, transform, diagnostics
