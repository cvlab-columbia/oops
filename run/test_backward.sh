#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
#CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -i -m torch.distributed.launch --master_port=29519 --nproc_per_node=8 inner_model.py --batch_size 64 --fp16 -j 2 --opt_level O1 --resume_training --backward_predict --save_path test_results_back_nomeansub_fixedfns --checkpoint checkpoint_back_nomeansub --test_only --save_test
