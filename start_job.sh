export FLAGS_call_stack_level=2
export PADDLE_WITH_GLOO=0
# export CUDA_VISIBLE_DEVICES=0
export GLOG_v=1


fleetrun --log_dir log train.py \
    --use_amp=True \
    --use_recompute=True \
    --run_id=Recompute_Bert_Large
  
