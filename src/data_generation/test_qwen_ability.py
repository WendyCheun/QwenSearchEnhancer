import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
from typing import List, Dict, Any, Union

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """使用模型生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, questions, output_dir):
    """评估模型在问题列表上的表现"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for idx, question in enumerate(tqdm(questions)):
        query = question["question"]
        
        # 构建提示
        prompt = f"问题: {query}\n回答:"
        
        # 生成回复
        response = generate_response(model, tokenizer, prompt)
        
        # 保存结果
        result = {
            "id": idx,
            "query": query,
            "response": response
        }
        results.append(result)
    
    # 保存所有结果到文件
    with open(os.path.join(output_dir, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，结果已保存到 {os.path.join(output_dir, 'baseline_results.json')}")
    return results

def main():
    # 模型路径
    model_name = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
    
    # 输出目录
    output_dir = "/remote-home1/wxzhang/QwenSearchEnhancer/data"
    
    # 加载模型和分词器
    print(f"正在加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 直接定义问题列表
    questions = [
        {"question": "在当前法律与伦理框架下，媒体如何平衡报道责任与公众知情权？"},
        {"question": "如何通过可持续的传播策略，推动媒体与公众实现更深层次的社会责任与互动？"},
        {"question": "在全球范围内，如何通过公共政策促进可持续发展目标的同时确保社会公平和效率的平衡？"},
        {"question": "在数据驱动的决策中，如何在保护公民隐私和促进公共政策制定之间找到最佳解决方案？"},
        {"question": "在应对气候变化的背景下，如何通过公共政策促进绿色技术的创新和应用，同时确保这些政策的可操作性和可行性？"},
        {"question": "在全球化背景下，公共政策如何应对技术带来的新的治理挑战？"}
    ]
    
    # 评估模型
    evaluate_model(model, tokenizer, questions, output_dir)

if __name__ == "__main__":
    main()