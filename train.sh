export VAE_MODEL="models/sd-vae-ft-mse/"
export DATASET="datasets"
export UNET_CONFIG="config/musetalk.json"

CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 train.py \
  --mixed_precision="fp16" \
  --unet_config_file=$UNET_CONFIG \
  --pretrained_model_name_or_path=$VAE_MODEL \
  --data_root=$DATASET \
  --train_batch_size=30 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=200000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir="musetalk_all" \
  --val_out_dir='musetalk_all_val' \
  --testing_speed \
  --checkpointing_steps=20000 \
  --validation_steps=20000 \
  --reconstruction \
  --resume_from_checkpoint="latest" \
  --use_audio_length_left=2 \
  --use_audio_length_right=2 \
  --whisper_model_type="tiny" \