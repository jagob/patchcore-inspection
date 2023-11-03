datapath=/home/jacob/data/ltd
loadpath=/home/jacob/code/patchcore-inspection/results/harbor_results

modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob
# modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob_1
savefolder=evaluated_results'/'$modelfolder

datasets=('harbor')
csv_path=/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test_1.csv

model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/harbor_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" --csv_path $csv_path harbor  $datapath
