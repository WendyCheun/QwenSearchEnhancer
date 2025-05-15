import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入自定义模型
import sys
sys.path.append("/remote-home1/wxzhang/QwenSearchEnhancer/src/training")
from search_model import create_search_aware_model

def load_test_data(test_file):
    """加载测试数据"""
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_samples = []
    for item in data["conversations"]:
        if "metadata" in item and "need_search" in item["metadata"]:
            # 提取用户问题
            user_messages = [turn for turn in item["conversations"] if turn["role"] == "user"]
            if user_messages:
                test_samples.append({
                    "question": user_messages[0]["content"],
                    "need_search": item["metadata"]["need_search"]
                })
    
    return test_samples

def evaluate_search_decision(model, tokenizer, test_samples, device="cuda"):
    """评估搜索判断能力"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="评估中"):
            # 准备输入
            prompt = f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # 获取模型输出
            outputs = model(**inputs)
            
            # 获取搜索判断预测
            if hasattr(outputs, "search_logits"):
                search_logit = outputs.search_logits[0].item()
                prediction = 1 if search_logit > 0 else 0
            else:
                # 如果模型没有搜索判断能力，默认预测为需要搜索
                prediction = 1
            
            predictions.append(prediction)
            labels.append(1 if sample["need_search"] else 0)
    
    # 计算评估指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "predictions": predictions,
        "labels": labels
    }

def main():
    parser = argparse.ArgumentParser(description="评估模型的搜索判断能力")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件")
    parser.add_argument("--output_file", type=str, default=None, help="评估结果输出文件")
    
    args = parser.parse_args()
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = create_search_aware_model(args.model_path, use_lora=False)
    
    # 将模型移至GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 加载测试数据
    test_samples = load_test_data(args.test_file)
    print(f"加载了 {len(test_samples)} 个测试样本")
    
    # 评估模型
    results = evaluate_search_decision(model, tokenizer, test_samples, device)
    
    # 打印评估结果
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print(f"混淆矩阵:\n{np.array(results['confusion_matrix'])}")
    
    # 保存评估结果
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main()