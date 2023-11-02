datapath=/home/jacob/data/ltd
loadpath=/home/jacob/code/patchcore-inspection/results/harbor_results

modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob_1
savefolder=evaluated_results'/'$modelfolder

datasets=('harbor')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/harbor_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" harbor  $datapath
