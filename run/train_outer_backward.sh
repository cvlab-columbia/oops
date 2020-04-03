#!/bin/bash
#OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
#CUDA_VISIBLE_DEVICES=0,1,2,3
python -i -m torch.distributed.launch --master_port=29558 --nproc_per_node=8 outer_model.py --batch_size 128 --fp16 -j 4 --opt_level O1 --save_test --save_path test_results_outer_back -c checkpoint_outer_back --pretrained_path checkpoint_back_nomeansub/model_best.pth --resume_training --backward_predict --save_test
