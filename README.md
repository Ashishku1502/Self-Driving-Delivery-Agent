# Self-Driving-Delivery-Agent

Based on the repository files, here's the **complete setup and how to run** the Self-Driving Delivery Agent project:

## 📦 Installation & Setup

### 1. **Environment Setup**
```bash
conda create --name=drivlme python=3.10
conda activate drivlme
```

### 2. **Clone & Install Dependencies**
```bash
git clone git@github.com:Ashishku1502/Self-Driving-Delivery-Agent.git
cd Self-Driving-Delivery-Agent
pip install -r requirements.txt
pip install -e .
```

---

## 🔧 Configuration

### 3. **Prepare LLaVA Model Weights**

You have two options:

**Option A: Download pre-trained weights directly**
- Download ready-made LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)

**Option B: Apply delta weights**
- Get the original LLaMA weights from [Hugging Face](https://huggingface.co/docs/transformers/main/model_doc/llama)
- Apply the delta:
```bash
python scripts/apply_delta.py \
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```

### 4. **Prepare Dataset**
Download the data from [this Dropbox link](https://www.dropbox.com/scl/fo/f429if26mveud6zcek54y/AEjkxF_DZ-DO87xJiOVkQTE?rlkey=shwm81sebtftttkflx8iqghxt&st=eux3itpx&dl=0) and extract:
```bash
tar -xf downloaded_file.tar
# Move to videos folder
mkdir -p videos
mv extracted_content/* videos/
```

---

## 🚀 Usage

### **Phase 1: Pretrain on BDD100K Dataset**
```bash
torchrun --nproc_per_node=4 --master_port 29001 drivlme/train/train_xformers.py \
          --model_name_or_path <Path to Llava> \
          --version v1 \
          --data_path datasets/bddx_pretrain.json \
          --video_folder videos/bdd100k_feats \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./DriVLMe_model_weights/bddx_pretrain_ckpt \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```
*Requires: 4 A40 GPUs*

### **Phase 2: Fine-tune on SDN Dataset**
```bash
deepspeed --master_port=29001 drivlme/train/train_xformers.py \
          --deepspeed ./scripts/zero2.json \
          --model_name_or_path <Path to Llava> \
          --pretrain_mm_mlp_adapter ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin \
          --version v1 \
          --lora_enable True \
          --lora_r 128 \
          --lora_alpha 256 \
          --data_path datasets/DriVLMe_sft_data.json \
          --video_folder videos/SDN_train_feats \
          --bf16 True \
          --output_dir ./model_path/DriVLMe \
          --num_train_epochs 3 \
          --per_device_train_batch_size 1 \
          --learning_rate 5e-5 \
          --warmup_ratio 0.03 \
          --tf32 True \
          --model_max_length 2048
```
*Requires: 4 A40 GPUs with DeepSpeed Zero2*

---

## 📊 Evaluation

### **Download Pre-trained Checkpoints**
[Download from Dropbox](https://www.dropbox.com/scl/fo/neqjdlhohygoa0wrv4uuy/AAjarkE6WY6sKt4LoAfyZ3c?rlkey=e0yvw6g1j8qqdp63vhgi0722d&st=tp2w6h3f&dl=0)

### **Run Inference for NfD Task**
```bash
python drivlme/single_video_inference_SDN.py \
    --model-name /path/to/LLaVA-7B-Lightening-v1-1/ \
    --projection_path ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin \
    --lora_path ./DriVLMe_model_weights/DriVLMe/ \
    --json_path datasets/SDN_test_actions.json \
    --video_root videos/SDN_test_videos/ \
    --out_path SDN_test_actions.json

python evaluation/physical_action_acc.py
```

### **Run Inference for RfN Task**
```bash
python drivlme/single_video_inference_SDN.py \
    --model-name /path/to/LLaVA-7B-Lightening-v1-1/ \
    --projection_path ./DriVLMe_model_weights/bddx_pretrain_ckpt/mm_projector.bin \
    --lora_path ./DriVLMe_model_weights/DriVLMe/ \
    --json_path datasets/SDN_test_conversations.json \
    --video_root videos/SDN_test_videos/ \
    --out_path SDN_test_conversations.json

python evaluation/diag_action_acc.py
```

---

## 💾 Key Requirements

- **Python**: 3.8+
- **GPUs**: 4 A40 GPUs (for training), 1 GPU minimum for inference
- **Key Dependencies**: DeepSpeed, PyTorch, Transformers, LLaVA
- **Distributed Training**: Supports DeepSpeed and Torch distributed training

For more details, check the full README in your repository!
