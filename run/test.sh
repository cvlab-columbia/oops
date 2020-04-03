#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=4,5,6,7 python -i -m torch.distributed.launch --master_port=29528 --nproc_per_node=4 binary_classify.py --batch_size 512 --fp16 -j 4 --opt_level O1 --save_test  --test_only --resume_training --save_path '/proj/vondrick/dave/results/sliding_weaksup/test_results_bincls_newborders_final' --sample_all_clips
#--resume_training
