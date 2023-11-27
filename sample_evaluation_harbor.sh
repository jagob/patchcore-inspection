datapath='/home/jacob/data/ltd/image_dataset'
# datapath=/home/jacob/data/harbor-mannequin
# datapath=/home/jacob/data/harbor-realfall

loadpath=/media/ehd2tb/models/patchcore/harbor

declare -a models=(
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0"
    "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random"
    "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1500random"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_resizeresize"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize"
    # "IM256_WR50_L3-4_P01_D1024-1024_PS-3_AN-1_S0_1200random"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize_imagenet"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-5_AN-1_S0_1200random"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-7_AN-1_S0_1200random"
    # "IM256_WR50_L2-3_P01_D1024-1024_PS-9_AN-1_S0_1200random"
    # "IM256_WR50_L2-3_P001_D1024-1024_PS-5_AN-1_S0_1200random"
    # "IM256_WR50_L2-3_P025_D1024-1024_PS-5_AN-1_S0_1200random"
)

declare -a test_csvs=(
    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test_1.csv"
    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test.csv"
    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_fast_moving_test.csv"
    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_near_edge_test.csv"
    "/home/jacob/code/harbor-synthetic/src/data/split/harbor_high_density_test.csv"
    "/home/jacob/code/harbor-synthetic/src/data/split/harbor_tampering_test.csv"

    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_mannequin_test.csv"
    # "/home/jacob/code/harbor-synthetic/src/data/split/harbor_realfall_test.csv"
)

datasets=('harbor')

for modelfolder in "${models[@]}" 
do
    savefolder=$loadpath'/'$modelfolder
    model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/harbor_'$dataset; done))
    dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

    for test_csv in "${test_csvs[@]}" 
    do
        echo "$test_csv"
        python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
        patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
        dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --test_csv $test_csv harbor $datapath
        echo ""
    done
done
