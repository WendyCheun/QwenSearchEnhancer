import os
import json
import random
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams

class QuestionGenerator:
    """基于种子数据生成新问题的类"""
    
    def __init__(self, model_path: str, seed_data_path: str):
        """
        初始化问题生成器
        
        Args:
            model_path: 用于生成问题的模型路径
            seed_data_path: 种子数据路径
        """
        self.model_path = model_path
        self.seed_data_path = seed_data_path
        
        # 加载种子数据
        self.seed_data = self._load_seed_data()
        
        # 初始化模型
        print(f"正在加载模型: {model_path}")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.8,  # 使用较高的温度以增加多样性
            top_p=0.95,
            max_tokens=512
        )
        print("模型加载完成")
    
    def _load_seed_data(self) -> List[Dict[str, Any]]:
        """加载种子数据"""
        if not os.path.exists(self.seed_data_path):
            raise FileNotFoundError(f"种子数据文件不存在: {self.seed_data_path}")
        
        data = []
        with open(self.seed_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"成功加载 {len(data)} 条种子数据")
        return data
    
    def _extract_topics(self) -> List[str]:
        """从种子数据中提取主题"""
        topics = set()
        
        for item in self.seed_data:
            if "instruction" in item:
                # 简单地将问题分割成词，并选择长度大于1的词作为潜在主题
                words = [w for w in item["instruction"].replace("？", "").replace("?", "").split() if len(w) > 1]
                topics.update(words)
            
            # 从搜索结果中提取更多主题
            if "metadata" in item and "search_results" in item["metadata"]:
                for result in item["metadata"]["search_results"]:
                    # 从每个搜索结果中提取前10个词作为潜在主题
                    words = result.split()[:10]
                    topics.update([w for w in words if len(w) > 1])
        
        return list(topics)
    
    def generate_questions(self, num_questions: int, output_path: str) -> List[str]:
        """
        生成新问题
        
        Args:
            num_questions: 要生成的问题数量
            output_path: 输出文件路径
            
        Returns:
            生成的问题列表
        """
        # 提取主题
        topics = self._extract_topics()
        print(f"从种子数据中提取了 {len(topics)} 个潜在主题")
        
        # 准备示例问题
        example_questions = [item["instruction"] for item in self.seed_data if "instruction" in item]
        
        generated_questions = []
        
        for _ in tqdm(range(num_questions), desc="生成问题"):
            # 随机选择3-5个主题
            selected_topics = random.sample(topics, min(random.randint(3, 5), len(topics)))
            
            # 随机选择3个示例问题
            selected_examples = random.sample(example_questions, min(3, len(example_questions)))
            
            # 构建提示
            prompt = f"""请基于以下主题和示例问题，生成一个新的、有深度的问题。问题应该是开放性的，需要搜索和思考才能回答。

主题: {', '.join(selected_topics)}

示例问题:
1. {selected_examples[0]}
2. {selected_examples[1]}
3. {selected_examples[2]}

请生成一个新问题:"""
            
            # 生成问题
            response = self.model.generate(prompt, self.sampling_params)
            question = response[0].outputs[0].text.strip()
            
            # 简单清理生成的问题
            if ":" in question:
                question = question.split(":", 1)[1].strip()
            if "\n" in question:
                question = question.split("\n", 1)[0].strip()
            if not question.endswith("?") and not question.endswith("？"):
                question += "？"
            
            generated_questions.append(question)
        
        # 保存生成的问题
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for question in generated_questions:
                    f.write(json.dumps({"question": question}, ensure_ascii=False) + "\n")
            print(f"已将 {len(generated_questions)} 个生成的问题保存至 {output_path}")
        
        return generated_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成新问题")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="模型路径")
    parser.add_argument("--seed_data", type=str, default="/remote-home1/wxzhang/QwenSearchEnhancer/data/synthetic/all_synthetic_data.jsonl",
                        help="种子数据路径")
    parser.add_argument("--num_questions", type=int, default=500,
                        help="要生成的问题数量")
    parser.add_argument("--output_path", type=str, default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/generated_questions.jsonl",
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 生成问题
    generator = QuestionGenerator(args.model_path, args.seed_data)
    generator.generate_questions(args.num_questions, args.output_path)