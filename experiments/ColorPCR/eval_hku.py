import argparse
import glob
import os.path as osp
import time

import numpy as np

from config import make_cfg
from geotransformer.engine import Logger
from geotransformer.utils.registration import compute_registration_error


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--method', default='lgr', choices=['lgr'])
    parser.add_argument('--verbose', action='store_true')
    return parser


def _mean(values):
    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()

    log_file = osp.join(cfg.log_dir, 'eval-hku-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    features_root = osp.join(cfg.feature_dir, args.benchmark)
    file_names = sorted(glob.glob(osp.join(features_root, '*', '*.npz')))
    if len(file_names) == 0:
        raise RuntimeError('No feature files found under "{}".'.format(features_root))

    rows = []
    success_rres = []
    success_rtes = []
    all_rres = []
    all_rtes = []
    for file_name in file_names:
        data = np.load(file_name)
        transform = data['transform']
        estimated_transform = data['estimated_transform']
        rre, rte = compute_registration_error(transform, estimated_transform)
        accepted = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
        all_rres.append(float(rre))
        all_rtes.append(float(rte))
        if accepted:
            success_rres.append(float(rre))
            success_rtes.append(float(rte))
        rows.append((file_name, accepted, float(rre), float(rte), float(data['overlap'])))
        if args.verbose:
            logger.info(
                '{} accepted={} overlap={:.3f} RRE={:.3f} RTE={:.3f}'.format(
                    file_name, int(accepted), float(data['overlap']), float(rre), float(rte)
                )
            )

    num_pairs = len(rows)
    num_success = sum(1 for _, accepted, _, _, _ in rows if accepted)
    rr = float(num_success) / float(num_pairs)
    message = (
        'HKU Registration, benchmark={}, pairs={}, success={}, fail={}, RR={:.6f}, '
        'mean_RRE_all={:.6f}, mean_RTE_all={:.6f}, mean_RRE_success={:.6f}, '
        'mean_RTE_success={:.6f}'
    ).format(
        args.benchmark,
        num_pairs,
        num_success,
        num_pairs - num_success,
        rr,
        _mean(all_rres),
        _mean(all_rtes),
        _mean(success_rres),
        _mean(success_rtes),
    )
    logger.critical(message)

    low_rows = [row for row in rows if 0.10 <= row[4] <= 0.30]
    if low_rows:
        low_success = sum(1 for _, accepted, _, _, _ in low_rows if accepted)
        low_rres = [row[2] for row in low_rows]
        low_rtes = [row[3] for row in low_rows]
        low_success_rres = [row[2] for row in low_rows if row[1]]
        low_success_rtes = [row[3] for row in low_rows if row[1]]
        logger.critical(
            (
                'HKU LowOverlap[0.10,0.30], pairs={}, success={}, fail={}, RR={:.6f}, '
                'mean_RRE_all={:.6f}, mean_RTE_all={:.6f}, mean_RRE_success={:.6f}, '
                'mean_RTE_success={:.6f}'
            ).format(
                len(low_rows),
                low_success,
                len(low_rows) - low_success,
                float(low_success) / float(len(low_rows)),
                _mean(low_rres),
                _mean(low_rtes),
                _mean(low_success_rres),
                _mean(low_success_rtes),
            )
        )


if __name__ == '__main__':
    main()
