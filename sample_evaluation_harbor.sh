datapath=/home/jacob/data/ltd
loadpath=/media/ehd2tb/models/patchcore/harbor

# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1500random
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_resizeresize
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize
# modelfolder=IM256_WR50_L3-4_P01_D1024-1024_PS-3_AN-1_S0_1200random
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize_imagenet
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-5_AN-1_S0_1200random
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-7_AN-1_S0_1200random
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-9_AN-1_S0_1200random
# modelfolder=IM256_WR50_L2-3_P001_D1024-1024_PS-5_AN-1_S0_1200random
modelfolder=IM256_WR50_L2-3_P025_D1024-1024_PS-5_AN-1_S0_1200random

test_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test.csv
# test_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test_1.csv

datasets=('harbor')
savefolder=$loadpath'/'$modelfolder
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/harbor_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --test_csv $test_csv harbor $datapath
