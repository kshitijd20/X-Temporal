T=`date +%m%d%H%M`
ROOT=../..
cfg=mmit.yaml
ad='/scratch/kshitijd/Algonauts2020/activations/slowfast101_MMIT_random'

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/extract_activations_random.py --config $cfg --activations_dir $ad | tee log.test.$T
