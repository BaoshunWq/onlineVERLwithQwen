set -x
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

python3 -m verl.trainer.main \
  config=my_configs/config.yaml \
  data.train_files=parquet_data_home/parquet_home/train.parquet \
  data.val_files=parquet_data_home/parquet_home/train.parquet \
  worker.actor.model.model_path=Qwen/Qwen2-VL-2B-Instruct \
  trainer.n_gpus_per_node=1\
  trainer.experiment_name=Qwen2B_LIBERO_AR_test \