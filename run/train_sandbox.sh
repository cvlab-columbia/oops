#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
NCCL_LL_THRESHOLD=0 CUDA_VISIBLE_DEVICES=4,5,6,7 python -i -m torch.distributed.launch --master_port=29528 --nproc_per_node=4 binary_classify.py --batch_size 4 --fp16 -j 2 --opt_level O1 -c '/proj/vondrick/dave/checkpoints/sliding_weaksup/bincls_linear' --linear_model
#--save_path '/proj/vondrick/dave/results/sliding_weaksup/test_results_bincls_linear'
#--resume_training
