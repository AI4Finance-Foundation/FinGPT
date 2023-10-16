export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed train_lora.py > train.log 2>&1