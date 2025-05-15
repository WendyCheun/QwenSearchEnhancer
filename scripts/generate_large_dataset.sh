#!/bin/bash

# 设置基础目录
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Base directory: $BASE_DIR"

# 创建必要的目录
mkdir -p "$BASE_DIR/data/generated"

# 运行大规模数据生成脚本
echo "Starting large-scale data generation..."
srun -p fnlp-4090d --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:2 python "$BASE_DIR/src/data_generation/generate_large_dataset.py" \
    --model_path "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --config_path "$BASE_DIR/configs/deepseek_config.yaml" \
    --seed_data "$BASE_DIR/data/synthetic/all_synthetic_data.jsonl" \
    --num_questions 5000 \
    --output_questions "$BASE_DIR/data/generated/generated_questions.jsonl" \
    --output_dir "$BASE_DIR/data/generated" \
    --use_slurm \
    --partition fnlp-4090d

echo "Large-scale data generation completed."