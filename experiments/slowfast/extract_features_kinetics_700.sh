T=`date +%m%d%H%M`
ROOT=../..
cfg=kinetics_700.yaml
ad = '/scratch/kshitijd/Algonauts2020/activations/slowfast101_Kinetics700'

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/extract_activations.py --config $cfg --activations_dir $ad | tee log.test.$T
