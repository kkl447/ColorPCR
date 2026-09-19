import argparse
import logging
import os
import sys

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
import time

from config import make_cfg

_BOOTSTRAP_CFG = make_cfg()
os.environ['CUDA_VISIBLE_DEVICES'] = _BOOTSTRAP_CFG.mac.cuda_visible_devices

import torch
from geotransformer.engine import SingleTester
from geotransformer.modules.geotransformer.mac_registration import MACRegistration
from geotransformer.modules.ops import index_select, point_to_node_partition
from geotransformer.utils.common import get_log_string
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.torch import release_cuda, to_cuda

from dataset import train_valid_data_loader
from loss import Evaluator
from model import create_model

def make_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('--dataset-root', default=None, help='override cfg.data.dataset_root')
    parser.add_argument('--metadata-dir', default=None, help='override cfg.data.metadata_dir')
    parser.add_argument('--num-pairs', type=int, default=None, help='evaluate only the first N val pairs')
    add_mac_args(parser)
    return parser


def add_mac_args(parser):
    parser.add_argument('--patch-topk', type=int, default=None)
    parser.add_argument('--max-corr', type=int, default=None, help='use -1 for no global correspondence cap')
    parser.add_argument('--edge-threshold', type=float, default=None)
    parser.add_argument('--dist-sigma-factor', type=float, default=None)
    parser.add_argument('--normal-sigma', type=float, default=None)
    parser.add_argument('--normal-k', type=int, default=None)
    parser.add_argument('--min-clique-size', type=int, default=None)
    parser.add_argument('--num-refinement-steps', type=int, default=None)
    parser.add_argument('--use-sog', dest='use_sog', action='store_true', default=None)
    parser.add_argument('--no-use-sog', dest='use_sog', action='store_false')
    parser.add_argument('--use-normal-fog', dest='use_normal_fog', action='store_true', default=None)
    parser.add_argument('--no-use-normal-fog', dest='use_normal_fog', action='store_false')
    parser.add_argument('--fallback-to-lgr', dest='fallback_to_lgr', action='store_true', default=None)
    parser.add_argument('--no-fallback-to-lgr', dest='fallback_to_lgr', action='store_false')
    parser.add_argument('--max-hypotheses', type=int, default=None)
    return parser


def apply_data_overrides(cfg, args):
    if args.dataset_root is not None:
        cfg.data.dataset_root = args.dataset_root
    if args.metadata_dir is not None:
        cfg.data.metadata_dir = args.metadata_dir


def apply_mac_overrides(cfg, args):
    override_names = (
        'patch_topk',
        'max_corr',
        'edge_threshold',
        'dist_sigma_factor',
        'normal_sigma',
        'normal_k',
        'min_clique_size',
        'num_refinement_steps',
        'use_sog',
        'use_normal_fog',
        'fallback_to_lgr',
        'max_hypotheses',
    )
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg.mac, name, value)
    if cfg.mac.max_corr is not None and cfg.mac.max_corr < 0:
        cfg.mac.max_corr = None


def format_mac_config(cfg):
    mac = cfg.mac
    return (
        'MAC config: snapshot_path={}, cuda_visible_devices={}, patch_topk={}, max_corr={}, '
        'edge_threshold={}, dist_sigma_factor={}, normal_sigma={}, normal_k={}, min_clique_size={}, '
        'num_refinement_steps={}, use_sog={}, use_normal_fog={}, fallback_to_lgr={}, '
        'max_hypotheses={}, use_weighted_svd={}, use_weighted_scoring={}'
    ).format(
        mac.snapshot_path,
        mac.cuda_visible_devices,
        mac.patch_topk,
        mac.max_corr,
        mac.edge_threshold,
        mac.dist_sigma_factor,
        mac.normal_sigma,
        mac.normal_k,
        mac.min_clique_size,
        mac.num_refinement_steps,
        mac.use_sog,
        mac.use_normal_fog,
        mac.fallback_to_lgr,
        mac.max_hypotheses,
        mac.use_weighted_svd,
        mac.use_weighted_scoring,
    )


def _safe_normalize(x, dim=-1, eps=1e-8):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


@torch.no_grad()
def estimate_normals_torch(points, normal_k):
    if points.shape[0] == 0:
        return points.new_zeros((0, 3))
    if points.shape[0] < 3:
        normals = points.new_zeros((points.shape[0], 3))
        normals[:, 2] = 1.0
        return normals

    k = min(int(normal_k), points.shape[0])
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
    knn_indices = dists.topk(k=k, largest=False, dim=1).indices
    patches = points[knn_indices]
    centered = patches - patches.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2).matmul(centered) / float(max(k - 1, 1))
    cov = 0.5 * (cov + cov.transpose(1, 2))
    cov = torch.where(torch.isfinite(cov), cov, torch.zeros_like(cov))
    _, eigvecs = torch.symeig(cov.detach().cpu(), eigenvectors=True)
    normals = eigvecs[:, :, 0].to(device=points.device, dtype=points.dtype)
    flip = ((-points) * normals).sum(dim=1) < 0.0
    normals = torch.where(flip[:, None], -normals, normals)
    return _safe_normalize(torch.where(torch.isfinite(normals), normals, torch.zeros_like(normals)))


@torch.no_grad()
def gather_node_corr_normals(output_dict, cfg):
    ref_normals_f = estimate_normals_torch(output_dict['ref_points_f'], cfg.mac.normal_k)
    src_normals_f = estimate_normals_torch(output_dict['src_points_f'], cfg.mac.normal_k)

    _, _, ref_knn_indices, _ = point_to_node_partition(
        output_dict['ref_points_f'], output_dict['ref_points_c'], cfg.model.num_points_in_patch
    )
    _, _, src_knn_indices, _ = point_to_node_partition(
        output_dict['src_points_f'], output_dict['src_points_c'], cfg.model.num_points_in_patch
    )

    ref_node_corr_knn_indices = ref_knn_indices[output_dict['ref_node_corr_indices']]
    src_node_corr_knn_indices = src_knn_indices[output_dict['src_node_corr_indices']]

    ref_padded_normals = torch.cat([ref_normals_f, torch.zeros_like(ref_normals_f[:1])], dim=0)
    src_padded_normals = torch.cat([src_normals_f, torch.zeros_like(src_normals_f[:1])], dim=0)
    ref_node_corr_knn_normals = index_select(ref_padded_normals, ref_node_corr_knn_indices, dim=0)
    src_node_corr_knn_normals = index_select(src_padded_normals, src_node_corr_knn_indices, dim=0)
    return ref_node_corr_knn_normals, src_node_corr_knn_normals


def scalar_to_python(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar_to_python(value[0])
    return value


def numeric_diagnostics(diagnostics):
    return {
        key: value
        for key, value in diagnostics.items()
        if key != 'mac_fallback_reason'
    }


def log_info_file_only(logger, message):
    raw_logger = getattr(logger, 'logger', None)
    if raw_logger is None:
        return
    record = raw_logger.makeRecord(
        raw_logger.name, logging.INFO, '', 0, message, args=(), exc_info=None
    )
    for handler in raw_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def format_progress_metrics(summary_dict):
    keys = ('RR', 'RRE', 'RTE', 'RMSE', 'IR', 'PIR')
    parts = []
    for key in keys:
        if key in summary_dict:
            parts.append('{} {:.3f}'.format(key, summary_dict[key]))
    return ' | '.join(parts)


def format_elapsed(seconds):
    seconds = int(max(float(seconds), 0.0))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return '{:02d}:{:02d}'.format(minutes, seconds)


def render_progress(current, total, elapsed, bar_width=36):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    ratio = float(current) / float(total)
    exact_fill = ratio * bar_width
    full_blocks = int(exact_fill)
    partial_index = int((exact_fill - full_blocks) * 8)
    partial_blocks = ('', '▏', '▎', '▍', '▌', '▋', '▊', '▉')
    partial = partial_blocks[partial_index] if full_blocks < bar_width else ''
    empty = ' ' * max(bar_width - full_blocks - (1 if partial else 0), 0)
    fill = '█' * full_blocks + partial
    bar = '\033[97m' + fill + '\033[0m' + empty
    return 'MAC val: {:3.0f}%|{}| {}/{} elapsed {}'.format(
        ratio * 100.0, bar, current, total, format_elapsed(elapsed)
    )


def update_progress(current, total, start_time):
    sys.stdout.write('\r' + render_progress(current, total, time.time() - start_time))
    sys.stdout.flush()


def finish_progress():
    sys.stdout.write('\n')
    sys.stdout.flush()


class Tester(SingleTester):
    def __init__(self, cfg, parser=None):
        super().__init__(cfg, parser=parser)
        self.cfg = cfg
        self.logger.info(format_mac_config(cfg))

        start_time = time.time()
        _, val_loader, neighbor_limits = train_valid_data_loader(cfg, distributed=False)
        loading_time = time.time() - start_time
        self.logger.info('Val data loader created: {:.3f}s collapsed.'.format(loading_time))
        self.logger.info('Calibrate neighbors: {}.'.format(neighbor_limits))
        self.register_loader(val_loader)

        model = create_model(cfg).cuda()
        self.register_model(model)
        self.evaluator = Evaluator(cfg).cuda()

        max_corr = cfg.mac.max_corr
        self.mac_registration = MACRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            cfg.backbone.init_voxel_size,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            patch_topk=cfg.mac.patch_topk,
            max_corr=max_corr,
            use_sog=cfg.mac.use_sog,
            edge_threshold=cfg.mac.edge_threshold,
            dist_sigma_factor=cfg.mac.dist_sigma_factor,
            min_clique_size=cfg.mac.min_clique_size,
            normal_sigma=cfg.mac.normal_sigma,
            use_normal_fog=cfg.mac.use_normal_fog,
            num_refinement_steps=cfg.mac.num_refinement_steps,
            max_hypotheses=cfg.mac.max_hypotheses,
            use_weighted_svd=cfg.mac.use_weighted_svd,
            use_weighted_scoring=cfg.mac.use_weighted_scoring,
        ).cuda()

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        matching_scores = output_dict['matching_scores']
        if not self.mac_registration.use_dustbin:
            matching_scores = matching_scores[:, :-1, :-1]

        ref_normals = None
        src_normals = None
        if self.mac_registration.use_normal_fog:
            ref_normals, src_normals = gather_node_corr_normals(output_dict, self.cfg)

        (
            mac_ref_corr_points,
            mac_src_corr_points,
            mac_corr_scores,
            mac_transform,
            mac_diagnostics,
        ) = self.mac_registration(
            output_dict['ref_node_corr_knn_points'],
            output_dict['src_node_corr_knn_points'],
            output_dict['ref_node_corr_knn_masks'],
            output_dict['src_node_corr_knn_masks'],
            matching_scores,
            output_dict.get('node_corr_scores', torch.ones(output_dict['ref_node_corr_indices'].shape[0], device=matching_scores.device, dtype=matching_scores.dtype)),
            ref_knn_normals=ref_normals,
            src_knn_normals=src_normals,
        )

        output_dict['lgr_estimated_transform'] = output_dict['estimated_transform']
        output_dict['mac_ref_corr_points'] = mac_ref_corr_points
        output_dict['mac_src_corr_points'] = mac_src_corr_points
        output_dict['mac_corr_scores'] = mac_corr_scores
        output_dict['mac_diagnostics'] = mac_diagnostics

        if mac_transform is not None:
            output_dict['estimated_transform'] = mac_transform
        elif not self.cfg.mac.fallback_to_lgr:
            raise RuntimeError('MAC failed without LGR fallback: {}'.format(mac_diagnostics['mac_fallback_reason']))

        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        result_dict.update(numeric_diagnostics(output_dict['mac_diagnostics']))
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = scalar_to_python(data_dict['scene_name'])
        ref_frame = scalar_to_python(data_dict['ref_frame'])
        src_frame = scalar_to_python(data_dict['src_frame'])
        reason = output_dict['mac_diagnostics'].get('mac_fallback_reason', '')
        message = '{}, id0: {}, id1: {}'.format(scene_name, ref_frame, src_frame)
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', mac_fallback_reason: {}'.format(reason if reason else 'none')
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        message += ', nMacCorr: {}'.format(output_dict['mac_corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        reason = output_dict['mac_diagnostics'].get('mac_fallback_reason', '')
        if reason:
            log_info_file_only(self.logger, 'MAC fallback at iter {}: {}'.format(iteration, reason))

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        if self.args.num_pairs is not None:
            total_iterations = min(total_iterations, int(self.args.num_pairs))
        last_progress_percent = -1
        progress_start_time = time.time()
        update_progress(0, total_iterations, progress_start_time)
        for iteration, data_dict in enumerate(self.test_loader):
            if self.args.num_pairs is not None and iteration >= int(self.args.num_pairs):
                break
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += ', {}'.format(timer.tostring())
            log_info_file_only(self.logger, message)
            progress_percent = int(100.0 * self.iteration / max(total_iterations, 1))
            if progress_percent != last_progress_percent or self.iteration == total_iterations:
                update_progress(self.iteration, total_iterations, progress_start_time)
                last_progress_percent = progress_percent
            torch.cuda.empty_cache()
        finish_progress()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)


def main():
    pre_parser = make_parser(add_help=False)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = make_cfg()
    apply_data_overrides(cfg, pre_args)
    apply_mac_overrides(cfg, pre_args)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    if not any(arg in sys.argv for arg in ('--snapshot', '--test_epoch', '--test_iter', '-h', '--help')):
        sys.argv.extend(['--snapshot', cfg.mac.snapshot_path])
    tester = Tester(cfg, parser=parser)
    tester.run()


if __name__ == '__main__':
    main()
