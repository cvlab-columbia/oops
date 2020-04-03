#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
#CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=0 python -i -m torch.distributed.launch --master_port=29530 --nproc_per_node=1 outer_model.py --batch_size 4 --fp16 -j 0 --opt_level O1 --save_test --resume_training
