import argparse
import yaml
from easydict import EasyDict
import torch

from x_temporal.interface.temporal_helper import TemporalHelper
from x_temporal.utils.multiprocessing import mrun


parser = argparse.ArgumentParser(description='X-Temporal')
parser.add_argument('--config', type=str, help='the path of config file')
parser.add_argument("--shard_id", help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0, type=int)
parser.add_argument("--num_shards", help="Number of shards using by the job",
        default=1, type=int)
parser.add_argument("--init_method", help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999", type=str)
parser.add_argument('--dist_backend', default='nccl', type=str)
parser.add_argument('--activations_dir', default='/scratch/kshitijd/Algonauts2020/activations/', type=str)

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
