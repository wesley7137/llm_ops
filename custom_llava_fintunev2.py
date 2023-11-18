import os
import deepspeed
from llava.train import train_mem
from llava.model.builder import load_pretrained_model

# Set CUDA to only see GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Define your training parameters
model_checkpoint = "liuhaotian/llava-v1.5-7b"  # Change this to the checkpoint you want to use
data_path = "/root/LLaVA/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json"  # Path to the ScienceQA dataset
image_folder = "/root/LLaVA/playground/data/train"  # Path to the images folder
output_dir = "/root/LLaVA/model_outputs"

# Training function
def train():
    # Load the pretrained model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_checkpoint,
        model_base=None,
        model_name=get_model_name_from_path(model_checkpoint)
    )

    # Start training
    train_mem.train(
        model_name_or_path=model_checkpoint,
        data_path=data_path,
        image_folder=image_folder,
        output_dir=output_dir,
        # Add additional training parameters here
    )

    # Evaluate the model after training
    evaluate(model, data_path, image_folder)

# DeepSpeed Configuration
deepspeed_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}

# Fine-tuning command
fine_tuning_command = (
    "deepspeed llava/train/train_mem.py "
    "--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 "
    "--deepspeed ./scripts/zero3.json "
    "--model_name_or_path liuhaotian/llava-v1.5-7b "
    "--version v1 "
    "--data_path /root/LLaVA/ScienceQA/data/scienceqa/llava_train_QCM-LEA.json "
    "--image_folder /root/LLaVA/playground/data/train "
    "--vision_tower openai/clip-vit-large-patch14-336 "
    "--mm_projector_type mlp2x_gelu "
    "--mm_vision_select_layer -2 "
    "--mm_use_im_start_end False "
    "--mm_use_im_patch_token False "
    "--image_aspect_ratio pad "
    "--group_by_modality_length True "
    "--fp16 True "
    "--output_dir /root/LLaVA/model_outputs "
    "--num_train_epochs 1 "
    "--per_device_train_batch_size 4 "
    "--per_device_eval_batch_size 2 "
    "--gradient_accumulation_steps 4 "
    "--evaluation_strategy 'no' "
    "--save_strategy 'steps' "
    "--save_steps 100 "
    "--save_total_limit 1 "
    "--learning_rate 2e-4 "
    "--weight_decay 0. "
    "--warmup_ratio 0.03 "
    "--lr_scheduler_type 'cosine' "
    "--logging_steps 1 "
    "--tf32 True "
    "--model_max_length 2048 "
    "--gradient_checkpointing True "
    "--dataloader_num_workers 4 "
    "--lazy_preprocess True "
    "--report_to wandb"
)

# Execute the training command
os.system(fine_tuning_command)

if __name__ == "__main__":
    train()


