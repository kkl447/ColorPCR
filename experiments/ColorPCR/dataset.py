import os.path as osp
import pickle

import numpy as np
import torch.utils.data
from skimage import color

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


class HKUMarsColorPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        metadata_dir='metadata_amtown_valtest',
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1.0,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.subset = subset
        self.metadata_dir = metadata_dir
        self.point_limit = point_limit
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_rotation = augmentation_rotation

        metadata_path = osp.join(self.dataset_root, self.metadata_dir, '{}.pkl'.format(subset))
        with open(metadata_path, 'rb') as f:
            self.metadata_list = pickle.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = np.load(osp.join(self.dataset_root, file_name))
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if np.random.rand() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points = ref_points + (
            np.random.rand(ref_points.shape[0], 3) - 0.5
        ) * self.augmentation_noise
        src_points = src_points + (
            np.random.rand(src_points.shape[0], 3) - 0.5
        ) * self.augmentation_noise
        return ref_points, src_points, get_transform_from_rotation_translation(rotation, translation)

    def _split_points_and_hsv(self, raw_points):
        points = raw_points[:, :3].astype(np.float32)
        if raw_points.shape[1] < 6:
            raise ValueError('HKU RGB data requires XYZRGB point clouds with at least 6 columns.')
        rgb = np.clip(raw_points[:, 3:6], 0.0, 1.0).astype(np.float32)
        hsv = color.rgb2hsv(rgb).astype(np.float32)
        return points, hsv

    def __getitem__(self, index):
        metadata = self.metadata_list[index]
        ref_raw_points = self._load_point_cloud(metadata['pcd0'])
        src_raw_points = self._load_point_cloud(metadata['pcd1'])

        ref_points, ref_hsv = self._split_points_and_hsv(ref_raw_points)
        src_points, src_hsv = self._split_points_and_hsv(src_raw_points)
        transform = metadata['transform'].astype(np.float32)

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(
                ref_points, src_points, transform
            )

        data_dict = {
            'index': index,
            'scene_name': metadata.get('seq', str(metadata.get('seq_id', 0))),
            'ref_frame': metadata['frame0'],
            'src_frame': metadata['frame1'],
            'overlap': metadata['overlap'],
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'ref_feats': np.ones((ref_points.shape[0], 1), dtype=np.float32),
            'src_feats': np.ones((src_points.shape[0], 1), dtype=np.float32),
            'ref_hsv': ref_hsv,
            'src_hsv': src_hsv,
            'transform': transform.astype(np.float32),
        }
        return data_dict


def _make_dataset(
    cfg,
    subset,
    point_limit,
    use_augmentation=False,
):
    if cfg.data.dataset_type == 'hku_mars_rgb':
        return HKUMarsColorPairDataset(
            cfg.data.dataset_root,
            subset,
            metadata_dir=cfg.data.metadata_dir,
            point_limit=point_limit,
            use_augmentation=use_augmentation,
            augmentation_noise=cfg.train.augmentation_noise,
            augmentation_rotation=cfg.train.augmentation_rotation,
        )
    return ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        subset,
        point_limit=point_limit,
        use_augmentation=use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        extra_channel='hsv',
    )


def train_valid_data_loader(cfg, distributed):
    train_dataset = _make_dataset(
        cfg,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = _make_dataset(
        cfg,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    train_dataset = _make_dataset(
        cfg,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = _make_dataset(
        cfg,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
