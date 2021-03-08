import argparse
import yaml
from easydict import EasyDict
import torch
import os
from PIL import Image
from x_temporal.interface.temporal_helper_local import TemporalHelper
from x_temporal.utils.multiprocessing import mrun
from x_temporal.core.transforms import *
from x_temporal.utils.dataset_helper import get_val_crop_transform, get_dataset, shuffle_dataset
from decord import VideoReader
from decord import cpu


def get_model():
    parser = argparse.ArgumentParser(description='X-Temporal')
    parser.add_argument('--config', type=str, default = "D:/Projects/Algonauts2020/X-Temporal/experiments/slowfast/mmit_local.yaml" ,help='the path of config file')
    parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
            default=0, type=int)
    parser.add_argument("--num_shards", help="Number of shards using by the job",
            default=1, type=int)
    parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999", type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['config'])
    print(config)
    temporal_helper = TemporalHelper(config, inference_only=True)
    return temporal_helper.model

def get_indices(config,num_frames):

    sample_range = config.dataset.num_segments * config.dataset.dense_sample_rate
    sample_pos = max(1, 1 + num_frames - sample_range)
    t_stride = config.dataset.dense_sample_rate
    if config.evaluate.temporal_samples == 1:
        start_idx = 0 if sample_pos == 1 else sample_pos // 2
        offsets = [
            (idx * t_stride + start_idx) %
            num_frames for idx in range(
            config.dataset.num_segments)]
    return np.array(offsets) + 1




def get_video_input(file):
    parser = argparse.ArgumentParser(description='X-Temporal')
    parser.add_argument('--config', type=str, default = "D:/Projects/Algonauts2020/X-Temporal/experiments/slowfast/mmit_local.yaml" ,help='the path of config file')
    parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
            default=0, type=int)
    parser.add_argument("--num_shards", help="Number of shards using by the job",
            default=1, type=int)
    parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999", type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config['config'])

    images = list()
    vr = VideoReader(file, ctx=cpu(0))
    print(len(vr))
    num_frames = len(vr)
    indices = get_indices(config,num_frames)
    print(indices)
    for seg_ind in indices:
        try:
            images.append(Image.fromarray(vr[seg_ind-1].asnumpy()))
        except Exception as e:
            images.append(Image.fromarray(vr[0].asnumpy()))

    #defining transformations
    dargs = config.dataset
    if dargs.modality == 'RGB':
        data_length = 1
    spatial_crops = 1
    temporal_samples = 1
    normalize = GroupNormalize(dargs.input_mean, dargs.input_std)
    crop_aug = get_val_crop_transform(config.dataset, spatial_crops)
    transform = torchvision.transforms.Compose([
        GroupScale(int(dargs.scale_size)),
        crop_aug,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
        ConvertDataFormat(config.net.model_type),
    ])



    process_data = transform(images)
    process_data = process_data.unsqueeze(0)
    # more transformations
    isizes = process_data.shape
    dup_samples = 1

    process_data = process_data.view(
        isizes[0], isizes[1], dup_samples, -1, isizes[3], isizes[4]
            )
    process_data = process_data.permute(0, 2, 1, 3, 4, 5).contiguous()
    process_data = process_data.view(isizes[0] * dup_samples, isizes[1], -1, isizes[3], isizes[4])

    return process_data

def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['config'])
    temporal_helper = TemporalHelper(config, inference_only=True)
    if not os.path.exists(args.activations_dir):
        os.makedirs(args.activations_dir)
    temporal_helper.extract_activations(args.activations_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("forkserver")
    main()
