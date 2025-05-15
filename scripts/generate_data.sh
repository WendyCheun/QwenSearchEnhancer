#!/bin/bash

# 设置基础目录
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Base directory: $BASE_DIR"

# 创建必要的目录
mkdir -p "$BASE_DIR/data/raw"
mkdir -p "$BASE_DIR/data/synthetic"

# 准备示例问题
if [ ! -f "$BASE_DIR/data/raw/questions.jsonl" ]; then
    echo "Creating sample questions file..."
    cat > "$BASE_DIR/data/raw/questions.jsonl" << EOL
{"question": "量子计算机的工作原理是什么？"}
{"question": "2023年世界经济论坛的主要议题有哪些？"}
{"question": "如何评价ChatGPT对就业市场的影响？"}
{"question": "气候变化对全球粮食安全有什么影响？"}
{"question": "最新的人工智能研究突破有哪些？"}
{"question": "如何解决大城市交通拥堵问题？"}
{"question": "区块链技术在供应链管理中的应用案例？"}
{"question": "太空探索的最新进展是什么？"}
{"question": "如何提高远程工作的效率？"}
{"question": "元宇宙概念对未来社交媒体的影响？"}
{"question": "生成式AI对创意产业的影响是什么？"}
{"question": "全球芯片短缺的原因和影响是什么？"}
{"question": "可再生能源在解决能源危机中的作用？"}
{"question": "数字货币对传统金融体系的挑战有哪些？"}
{"question": "大型语言模型的伦理问题有哪些？"}
{"question": "未来十年最有前景的职业方向是什么？"}
{"question": "如何有效应对网络安全威胁？"}
{"question": "人工智能在医疗诊断中的应用进展？"}
{"question": "全球供应链重构的趋势和影响？"}
{"question": "智能城市建设的关键技术和挑战？"}
EOL
    echo "Sample questions created."
fi

# 运行数据生成脚本
echo "Starting data generation..."
#python "$BASE_DIR/src/data_generation/deepseek_inference.py" \
srun -p fnlp-4090 --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:1 python "$BASE_DIR/src/data_generation/deepseek_inference.py" \
    --model_path "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --config_path "$BASE_DIR/configs/deepseek_config.yaml" \
    --input_questions "$BASE_DIR/data/raw/questions.jsonl" \
    --output_dir "$BASE_DIR/data/synthetic" \
    --samples_per_type 100

echo "Data generation completed."