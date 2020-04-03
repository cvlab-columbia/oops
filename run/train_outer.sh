#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=0,1,2,3 python -i -m torch.distributed.launch --master_port=29550 --nproc_per_node=4 outer_model.py --batch_size 128 --fp16 -j 4 --opt_level O1 --save_test --save_path test_results_outer -c checkpoint_outer --pretrained_path pretrained/inner_model_meansub.pth --resume_training --subtract_mean
