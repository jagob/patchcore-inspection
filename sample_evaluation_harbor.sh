datapath=/home/jacob/data/ltd
loadpath=/media/ehd2tb/models/patchcore/harbor

# modelfolder=IM256-3_P01_D1024-1024_PS-3_AN-1_S0_jacob
# modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob_1
modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob

csv_path=/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test_1.csv

datasets=('harbor')
savefolder=$loadpath'/'$modelfolder
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/harbor_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --csv_path $csv_path harbor $datapath
