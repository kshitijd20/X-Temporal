T=`date +%m%d%H%M`
ROOT=../..
cfg=kinetics_600.yaml
ad='/scratch/kshitijd/Algonauts2020/activations/slowfast50_Kinetics600'

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/extract_activations.py --config $cfg --activations_dir $ad | tee log.test.$T
