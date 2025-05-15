import os
import argparse
import subprocess
from pathlib import Path

def ensure_dir(directory):
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="使用fake_search为问题生成回答")
    parser.add_argument("--model_path", type=str, 
                        default="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="模型路径")
    parser.add_argument("--config_path", type=str, 
                        default="/remote-home1/wxzhang/QwenSearchEnhancer/configs/deepseek_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--input_questions", type=str, 
                        default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/direct_generated_questions.jsonl",
                        help="输入问题文件路径")
    parser.add_argument("--output_dir", type=str, 
                        default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset",
                        help="输出数据目录")
    parser.add_argument("--samples_per_type", type=int, default=100,
                        help="每种类型要处理的样本数量")
    parser.add_argument("--use_slurm", action="store_true",
                        help="是否使用Slurm调度系统")
    parser.add_argument("--partition", type=str, default="fnlp-4090d",
                        help="Slurm分区名称")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 调用数据生成脚本
    cmd = [
        "python", 
        os.path.join("/remote-home1/wxzhang/QwenSearchEnhancer/src/data_generation", "deepseek_inference_2.0.py"),
        "--model_path", args.model_path,
        "--config_path", args.config_path,
        "--input_questions", args.input_questions,
        "--output_dir", args.output_dir,
        "--samples_per_type", str(args.samples_per_type),
        "--fake"  # 使用假搜索
    ]
    
    if args.use_slurm:
        cmd = ["srun", "-p", args.partition, "--cpus-per-task=4", "--mem-per-cpu=4G", "--gres=gpu:2"] + cmd
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"数据生成完成，已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()