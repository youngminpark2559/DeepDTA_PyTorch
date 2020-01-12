# conda activate py36gputorch100 && \
# cd /mnt/external_disk/Code_projects/Drug_development/DeepDTA_PyTorch/V_00001/prj_root

# ================================================================================
# Train
# rm e.l && python main.py \
# --start_mode="train" \
# --epoch=9 \
# --batch_size=2 \
# --task_mode="train" \
# --train_method="deepdta" \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# dir_where_text_file_for_image_paths_is_in="./../Data"

# --start_mode="preanalyze_data" \
# --start_mode="train" \
# --start_mode="classify_docs" \

# --network_type="ResNet50_CBAM" \
# --network_type="Scikit_Learn_SVM" \

# --train_method="Scikit_Learn_SVM" \
# --train_method="xgboost" \
# --train_method="xgboost_one_hot" \

# --classify_docs_method="use_vocab" \

# ================================================================================
# Validataion
# rm e.l && python main.py \
# --task_mode="validation" \
# --train_method="train_by_transfer_learning_using_resnet" \
# --use_saved_model_for_continuous_train=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=5 \
# --batch_size=30 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=10 \
# --leapping_term_when_saving_model_after_batch=10 \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# Submission
# rm e.l && python main.py \
# --task_mode="submission" \
# --train_method="train_by_transfer_learning_using_resnet" \
# --use_saved_model_for_continuous_train=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/tumor_pretrained_resnet50_cbam.pth" \
# --epoch=5 \
# --batch_size=200 \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --dir_where_text_file_for_image_paths_is_in="/mnt/1T-5e7/mycodehtml/bio_health/Kaggle_histopathologic-cancer-detection/Data" \
# --use_multi_gpu=False \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --optimizer=None \
# --network_type="ResNet50_CBAM" \
# --leapping_term_when_displaying_loss=10 \
# --leapping_term_when_saving_model_after_batch=10 \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import datetime
import sys,os,copy,argparse

import torch

from src.api_argument import argument_api_module as argument_api_module
from src.train_script import train_by_deepdta as train_by_deepdta

# ================================================================================
def start_train(args):
  if args.train_method=="deepdta":
    train_by_deepdta.train(args)
  else:
    pass

# ================================================================================
if __name__=="__main__":
  # c argument_api: instance of Argument_API_class
  argument_api=argument_api_module.Argument_API_class()

  # c args: member attribute from argument_api
  args=argument_api.args
  # print("args",args)
  # Namespace(
  #   batch_size='2', 
  #   epoch='9', 
  #   start_mode='classify_docs', 
  #   task_mode='train', 
  #   train_method='xgboost_one_hot')

  # ================================================================================
  if args.start_mode=="train":
    start_train(args)
