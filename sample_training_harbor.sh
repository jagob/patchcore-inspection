datapath=/home/jacob/data/ltd
log_project=/media/ehd2tb/models/patchcore/harbor
datasets=('harbor')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection Harborfront
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC:0.8636 , Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_0010.csv
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9554 , Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9548 , Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1500random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1500 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9101 , Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_resizeresize
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9431, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # using imagenet mean and std normalization
# # Performance: Instance AUROC:0.9272 , Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_1200random_normalize_imagenet
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1500 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Using layer 3 and 4, for video data, according to paper
# # Performance: Instance AUROC:0.9006, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L3-4_P01_D1024-1024_PS-3_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer3 -le layer4 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9808, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-5_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 5 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9825, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-7_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 7 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.968, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-9_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 9 sampler -p 0.1 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# # Performance: Instance AUROC:0.9827, Pixelwise AUROC: 0, PRO: 0
# modelfolder=IM256_WR50_L2-3_P001_D1024-1024_PS-5_AN-1_S0_1200random
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
# python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 5 sampler -p 0.01 approx_greedy_coreset \
# dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath

# Performance: Instance AUROC:0.9844, Pixelwise AUROC: 0, PRO: 0
modelfolder=IM256_WR50_L2-3_P025_D1024-1024_PS-5_AN-1_S0_1200random
train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_1000.csv  # 1200 random samples
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 5 sampler -p 0.25 approx_greedy_coreset \
dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath
