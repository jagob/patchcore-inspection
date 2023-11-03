datapath=/home/jacob/data/ltd
log_project=/media/ehd2tb/models/patchcore/harbor
datasets=('harbor')

############# Detection Harborfront
modelfolder=IM256_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_jacob
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group $modelfolder --log_project $log_project results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 256 "${dataset_flags[@]}" harbor $datapath
