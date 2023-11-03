datapath=/home/jacob/data/ltd
log_project=/media/ehd2tb/models/patchcore/harbor
datasets=('harbor')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection Harborfront
# IM256:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0
# Performance: Instance AUROC: 0.992, Pixelwise AUROC: 0.981, PRO: 0.944
modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_0001.csv
# train_csv=/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_0010.csv
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset \
dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" --train_csv $train_csv harbor $datapath
