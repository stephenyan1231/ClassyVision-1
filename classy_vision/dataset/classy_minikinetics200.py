#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import os
from typing import Any, Callable, Dict, Optional

import torch
from torchvision.datasets.video_utils import VideoClips

from classy_vision.dataset import register_dataset
from classy_vision.dataset.classy_video_dataset import ClassyVideoDataset
from classy_vision.dataset.transforms.util_video import (
    build_video_field_transform_default
)


class _MiniKinetics200Dataset:
    """TBD
    """

    def __init__(
        self,
        root,
        data_file,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        extension="mp4",
        transform=None,
        _precomputed_metadata=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
        _audio_channels=0,
    ) -> "MiniKinetics200Dataset":
        assert os.path.exists(data_file), f"Data file {data_file} is missing"
        self.samples = []
        with open(data_file, "r") as fp:
            for line in fp.readlines():
                video_id, class_name, class_label = line.strip().split(",")
                class_name = class_name.replace("_", " ")
                video_path = os.path.join(
                    root,
                    class_name,
                    f"{video_id}.{extension}",
                )
                if os.path.exists(video_path):
                    self.samples.append([video_path, int(class_label)])

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
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
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


@register_dataset("minikinetics200")
class MiniKinetics200Dataset(ClassyVideoDataset):
    """`Mini Kinetics-200 <https://github.com/s9xie/Mini-Kinetics-200>`_ is a
    subset of Kinetics400 dataset. It contains 200 classes, 400 videos per class
    in train set, and 25 videos per class in validation set.
    `Original publication <https://arxiv.org/abs/1712.04851>`_

    We assume videos are already trimmed to 10-second clip, and are stored in a
    folder.

    """

    def __init__(
        self,
        split: str,
        data_file: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform: Callable,
        num_samples: Optional[int],
        frames_per_clip: int,
        video_width: int,
        video_height: int,
        video_min_dimension: int,
        audio_samples: int,
        audio_channels: int,
        step_between_clips: int,
        frame_rate: Optional[int],
        clips_per_video: int,
        video_dir: str,
        extension: str,
        metadata_filepath: str,
    ):
        """The constructor of MiniKinetics200Dataset.

        Args:
            split: dataset split which can be either "train" or "test"
            data_file: a plain text file listing the videos in the split
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
            audio_samples: desired audio sample rate. If 0, keep original
                audio sample rate
            audio_channels: desire No. of audio channel. If 0, keep original audio
                channels
            step_between_clips: Number of frames between each clip.
            frame_rate: desired video frame rate. If None, keep
                orignal video frame rate.
            clips_per_video: Number of clips to sample from each video
            video_dir: path to video folder
            extension: File extension, such as "avi", "mp4". Only
                video matching the file extension are added to the dataset
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
            metadata = MiniKinetics200Dataset.load_metadata(
                metadata_filepath, video_dir=video_dir, update_file_path=True
            )

        dataset = _MiniKinetics200Dataset(
            video_dir,
            data_file,
            frames_per_clip,
            step_between_clips=step_between_clips,
            frame_rate=frame_rate,
            extension=extension,
            _precomputed_metadata=metadata,
            num_workers=torch.get_num_threads() // 2,  # heuristically use half threads
            _video_width=video_width,
            _video_height=video_height,
            _video_min_dimension=video_min_dimension,
            _audio_samples=audio_samples,
            _audio_channels=audio_channels,
        )
        metadata = dataset.metadata
        if metadata and not os.path.exists(metadata_filepath):
            MiniKinetics200Dataset.save_metadata(metadata, metadata_filepath)

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
    def from_config(cls, config: Dict[str, Any]) -> "MiniKinetics200Dataset":
        """Instantiates a MiniKinetics200Dataset from a configuration.

        Args:
            config: A configuration for a MiniKinetics200Dataset.
                See :func:`__init__` for parameters expected in the config.

        Returns:
            A MiniKinetics200Dataset instance.
        """
        required_args = ["split", "data_file", "metadata_file", "video_dir"]
        assert all(
            arg in config for arg in required_args
        ), f"The arguments {required_args} are all required."

        split = config["split"]
        audio_channels = config.get("audio_channels", 0)
        (
            transform_config,
            batchsize_per_replica,
            shuffle,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            step_between_clips,
            frame_rate,
            clips_per_video,
        ) = cls.parse_config(config)
        extension = config.get("extension", "mp4")

        transform = build_video_field_transform_default(transform_config, split)

        return cls(
            split,
            config["data_file"],
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            frames_per_clip,
            video_width,
            video_height,
            video_min_dimension,
            audio_samples,
            audio_channels,
            step_between_clips,
            frame_rate,
            clips_per_video,
            config["video_dir"],
            extension,
            config["metadata_file"],
        )
