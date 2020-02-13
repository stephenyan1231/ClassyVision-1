#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import os
from typing import Any, Callable, Dict, Optional

import torch
from torchvision.datasets.video_utils import VideoClips

from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_video_dataset import ClassyVideoDataset
from classy_vision.dataset.transforms.util_video import (
    build_video_field_transform_default
)


class _SomethingSomethingV2Dataset:
    """TBD
    """

    def __init__(
        self,
        video_dir,
        label_map_json,
        labels_json,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        transform=None,
        _precomputed_metadata=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
    ) -> "_SomethingSomethingV2Dataset":
        for data_file in [label_map_json, labels_json]:
            assert os.path.exists(data_file), f"Data file {data_file} is missing"

        with open(label_map_json, "r") as fp:
            label_map = json.load(fp)

        with open(labels_json, "r") as fp:
            samples = json.load(fp)
            self.samples = []
            for sample in samples:
                video_id = sample["id"]
                label = sample["template"].replace("[", "").replace("]", "")
                assert label in label_map, f"Unknown label: {label}"
                video_path = os.path.join(video_dir, f"{video_id}.webm")
                assert os.path.exists(video_path), f"{video_path} is missing"
                self.samples.append((video_path, int(label_map[label])))

        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label


@register_dataset("something_something_v2")
class SomethingSomethingV2Dataset(ClassyVideoDataset):
    """`SOMETHING-SOMETHING V2 dataset <https://20bn.com/datasets/something-something>`_
    is a large collection of densely-labeled video clips that show humans performing
    pre-defined basic actions with everyday objects.
    It contains 174 classes, 169K videos in train set, 25K videos in validation set,
    and 27K videos w/o labels in test set. We assume all videos are stored in a folder.

    """

    def __init__(
        self,
        split: str,
        label_map_json: str,
        labels_json: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Callable,
        num_samples: Optional[int],
        frames_per_clip: int,
        video_width: int,
        video_height: int,
        video_min_dimension: int,
        step_between_clips: int,
        frame_rate: Optional[int],
        clips_per_video: int,
        video_dir: str,
        metadata_filepath: str,
    ):
        """The constructor of SomethingSomethingV2Dataset.

        Args:
            split: dataset split which can be either "train" or "test"
            batchsize_per_replica: batch size per model replica
            shuffle: If true, shuffle the dataset
            transform: a dict where transforms video and audio data
            num_samples: if provided, it will subsample dataset
            frames_per_clip: the No. of frames in a video clip
            video_width: rescaled video width. If 0, keep original width
            video_height: rescaled video height. If 0, keep original height
            video_min_dimension: rescale video so that min(height, width) =
                video_min_dimension. If 0, keep original video resolution. Note
                only one of (video_width, video_height) and (video_min_dimension)
                can be set
            step_between_clips: Number of frames between each clip.
            frame_rate: desired video frame rate. If None, keep
                orignal video frame rate.
            clips_per_video: Number of clips to sample from each video
            video_dir: path to video folder
            metadata_filepath: path to the dataset meta data

        """
        # dataset metadata includes the path of video file, the pts of frames in
        # the video and other meta info such as video fps, duration, audio sample rate.
        # Users do not need to know the details of metadata. The computing, loading
        # and saving logic of metata are all handled inside of the dataset.
        # Given the "metadata_file" path, if such file exists, we load it as meta data.
        # Otherwise, we compute the meta data, and save it at "metadata_file" path.
        metadata = None
        if os.path.exists(metadata_filepath):
            metadata = SomethingSomethingV2Dataset.load_metadata(
                metadata_filepath, video_dir=video_dir, update_file_path=True
            )

        dataset = _SomethingSomethingV2Dataset(
            video_dir,
            label_map_json,
            labels_json,
            frames_per_clip,
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=metadata,
            num_workers=torch.get_num_threads() // 2,  # heuristically use half threads
            _video_width=video_width,
            _video_height=video_height,
            _video_min_dimension=video_min_dimension,
        )
        metadata = dataset.metadata
        if metadata and not os.path.exists(metadata_filepath):
            SomethingSomethingV2Dataset.save_metadata(metadata, metadata_filepath)

        super().__init__(
            dataset,
            split,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            clips_per_video,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SomethingSomethingV2Dataset":
        """Instantiates a SomethingSomethingV2Dataset from a configuration.

        Args:
            config: A configuration for a SomethingSomethingV2Dataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A SomethingSomethingV2Dataset instance.
        """
        required_args = [
            "split",
            "label_map_json",
            "labels_json",
            "metadata_file",
            "video_dir",
        ]
        assert all(
            arg in config for arg in required_args
        ), f"The arguments {required_args} are all required."

        split = config["split"]
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            _audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        ) = cls.parse_config(config)

        transform = build_video_field_transform_default(transform_config, split)

        return cls(
            split,
            config["label_map_json"],
            config["labels_json"],
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            step_between_clips,
            frame_rate,
            clips_per_video,
            config["video_dir"],
            config["metadata_file"],
        )

    @classmethod
    def load_metadata(
        cls,
        filepath: str,
        video_dir: Optional[str] = None,
        update_file_path: bool = False,
    ) -> Dict[str, Any]:
        """Load pre-computed video dataset meta data.

        Video dataset meta data computation takes minutes on small dataset and hours
        on large dataset, and thus is time-consuming. However, it only needs to be
        computed once, and can be saved into a file via :func:`save_metadata`.

        The format of meta data is defined in `TorchVision <https://github.com/
        pytorch/vision/blob/master/torchvision/datasets/video_utils.py#L131/>`_.

        For each video, meta data contains the video file path, presentation
        timestamps of all video frames, and video fps.

        Args:
            filepath: file path of pre-computed meta data
            video_dir: If provided, the folder where video files are stored.
            update_file_path: If true, replace the directory part of video file path
                in meta data with the actual video directory provided in `video_dir`.
                This is necessary for successsfully reusing pre-computed meta data
                when video directory has been moved and is no longer consitent
                with the full video file path saved in the meta data.
        """
        metadata = torch.load(filepath)
        if video_dir is not None and update_file_path:
            # video path in meta data can be computed in a different root video folder
            # If we use a different root video folder, we need to update the video paths
            assert os.path.exists(video_dir), "folder does not exist: %s" % video_dir
            for idx, video_path in enumerate(metadata["video_paths"]):
                # video path template is $VIDEO_DIR/$VIDEO_FILE
                _dirname, filename = os.path.split(video_path)
                metadata["video_paths"][idx] = os.path.join(video_dir, filename)
        return metadata
