import os
import json
import yaml
import time
import random
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from vllm import LLM, SamplingParams, LLMEngine
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generation.web_search import SearchTool
from utils.common import ensure_dir, load_jsonl, save_jsonl

class DeepSeekInference:
    """使用vllm进行DeepSeek模型推理的类"""
    
    def __init__(self, model_path: str, config_path: str = None, fake_search: bool = False):
        """
        初始化DeepSeek推理类
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径，包含推理参数
            fake_search: 是否使用假搜索
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        # 修改这一行，将fake参数改为use_fake_search参数
        self.search_tool = SearchTool(use_fake_search=fake_search)
        
        # 初始化vllm模型
        gpu_memory_utilization = self.config.get("gpu_memory_utilization", 0.9)
        tensor_parallel_size = self.config.get("tensor_parallel_size", 4)
        max_model_len = self.config.get("max_model_len", 131072)
        
        print(f"正在加载模型: {model_path}")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        print("模型加载完成")
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.9),
            max_tokens=self.config.get("max_tokens", 4096)
        )
        
        # 加载提示模板
        self.system_prompt = self.config.get("system_prompt", "")
        self.cot_prompt = self.config.get("cot_prompt", "")
        self.search_prompt = self.config.get("search_prompt", "")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 2,  # 修正键名
            "system_prompt": "你是一个智能助手，能够进行复杂思考并在需要时使用搜索工具。",
            "cot_prompt": "请对以下问题进行详细的思考，分析问题的各个方面，并给出合理的回答。",
            "search_prompt": "如果你需要最新信息或事实性知识，可以使用搜索工具。请先思考是否需要搜索，如果需要，请明确搜索关键词。"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                default_config.update(config)
        
        return default_config
    
    def generate_complex_thinking(self, questions: List[str], output_path: str) -> List[Dict[str, Any]]:
        """生成具有复杂思考能力的数据"""
        results = []
        
        for question in tqdm(questions, desc="生成复杂思考数据"):
            prompt = f"{self.system_prompt}\n\n{self.cot_prompt}\n\n用户问题: {question}\n\n请先思考，然后回答:"
            
            response = self.model.generate(prompt, self.sampling_params)
            answer = response[0].outputs[0].text.strip()
            
            result = {
                "instruction": question,
                "input": "",
                "output": answer,
                "type": "complex_thinking"
            }
            results.append(result)
        
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            save_jsonl(results, output_path)
        
        return results
    
    def generate_search_decision(self, questions: List[str], output_path: str) -> List[Dict[str, Any]]:
        """生成判断是否需要搜索的数据"""
        results = []
        
        for question in tqdm(questions, desc="生成搜索判断数据"):
            prompt = f"{self.system_prompt}\n\n{self.search_prompt}\n\n用户问题: {question}\n\n请思考是否需要使用搜索工具，并解释原因:"
            
            response = self.model.generate(prompt, self.sampling_params)
            answer = response[0].outputs[0].text.strip()
            
            # 提取是否需要搜索的决策
            need_search = "需要搜索" in answer or "使用搜索工具" in answer
            
            result = {
                "instruction": question,
                "input": "",
                "output": answer,
                "metadata": {"need_search": need_search},
                "type": "search_decision"
            }
            results.append(result)
        
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            save_jsonl(results, output_path)
        
        return results
    
    def generate_search_and_answer(self, questions: List[str], output_path: str) -> List[Dict[str, Any]]:
        """生成搜索并回答的数据"""
        results = []
        
        for question in tqdm(questions, desc="生成搜索并回答数据"):
            # 第一步：让模型决定搜索关键词
            keyword_prompt = f"{self.system_prompt}\n\n对于问题: {question}\n\n请提供适合搜索的关键词:"
            keyword_response = self.model.generate(keyword_prompt, self.sampling_params)
            search_keyword = keyword_response[0].outputs[0].text.strip()
            
            # 提取关键词（简单处理，实际应用中可能需要更复杂的提取逻辑）
            if "关键词:" in search_keyword:
                search_keyword = search_keyword.split("关键词:")[-1].strip()
            search_keyword = search_keyword.split("\n")[0].strip()
            
            # 第二步：执行搜索
            search_results = self.search_tool.search(search_keyword)
            formatted_results = self.search_tool.format_search_results(search_results)
            
            # 第三步：让模型基于搜索结果回答
            answer_prompt = f"{self.system_prompt}\n\n用户问题: {question}\n\n我已经搜索了相关信息，结果如下:\n\n{formatted_results}\n\n请基于这些搜索结果，进行思考并回答用户问题:"
            
            answer_response = self.model.generate(answer_prompt, self.sampling_params)
            final_answer = answer_response[0].outputs[0].text.strip()
            
            # 构建完整的对话流程
            full_output = f"思考：这个问题可能需要最新信息，我应该搜索一下。\n\n搜索关键词：{search_keyword}\n\n{formatted_results}\n\n基于搜索结果的回答：\n{final_answer}"
            
            result = {
                "instruction": question,
                "input": "",
                "output": full_output,
                "metadata": {
                    "search_keyword": search_keyword,
                    "search_results": search_results,
                    "final_answer": final_answer
                },
                "type": "search_and_answer"
            }
            results.append(result)
        
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            save_jsonl(results, output_path)
        
        return results
    
    def generate_all_data(self, input_questions_path: str, output_dir: str, sample_per_type: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """生成所有类型的数据"""
        # 加载问题
        if os.path.exists(input_questions_path):
            questions = load_jsonl(input_questions_path)
            if isinstance(questions[0], dict) and "question" in questions[0]:
                questions = [q["question"] for q in questions]
        else:
            # 如果没有提供问题文件，生成一些示例问题
            questions = [
                "量子计算机的工作原理是什么？",
                "2023年世界经济论坛的主要议题有哪些？",
                "如何评价ChatGPT对就业市场的影响？",
                "气候变化对全球粮食安全有什么影响？",
                "最新的人工智能研究突破有哪些？",
                "如何解决大城市交通拥堵问题？",
                "区块链技术在供应链管理中的应用案例？",
                "太空探索的最新进展是什么？",
                "如何提高远程工作的效率？",
                "元宇宙概念对未来社交媒体的影响？"
            ]
        
        # 确保输出目录存在
        ensure_dir(output_dir)
        
        # 为每种类型随机选择问题
        random.seed(42)  # 设置随机种子以确保可重复性
        
        all_questions = questions.copy()
        complex_questions = random.sample(all_questions, min(sample_per_type, len(all_questions)))
        search_decision_questions = random.sample(all_questions, min(sample_per_type, len(all_questions)))
        search_answer_questions = random.sample(all_questions, min(sample_per_type, len(all_questions)))
        
        # 生成各类数据
        results = {}
        
        complex_output_path = os.path.join(output_dir, "complex_thinking.jsonl")
        results["complex_thinking"] = self.generate_complex_thinking(complex_questions, complex_output_path)
        
        search_decision_output_path = os.path.join(output_dir, "search_decision.jsonl")
        results["search_decision"] = self.generate_search_decision(search_decision_questions, search_decision_output_path)
        
        search_answer_output_path = os.path.join(output_dir, "search_and_answer.jsonl")
        results["search_and_answer"] = self.generate_search_and_answer(search_answer_questions, search_answer_output_path)
        
        # 合并所有数据
        all_data = results["complex_thinking"] + results["search_decision"] + results["search_and_answer"]
        all_output_path = os.path.join(output_dir, "all_synthetic_data.jsonl")
        save_jsonl(all_data, all_output_path)
        
        print(f"数据生成完成，共生成 {len(all_data)} 条数据")
        print(f"- 复杂思考数据: {len(results['complex_thinking'])} 条")
        print(f"- 搜索判断数据: {len(results['search_decision'])} 条")
        print(f"- 搜索回答数据: {len(results['search_and_answer'])} 条")
        print(f"所有数据已保存至: {all_output_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="使用DeepSeek-R1-Distill-Qwen-7B生成合成数据")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="DeepSeek模型路径")
    parser.add_argument("--config_path", type=str, default="../../configs/deepseek_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--input_questions", type=str, default="../../data/raw/questions.jsonl",
                        help="输入问题文件路径")
    parser.add_argument("--output_dir", type=str, default="../../data/synthetic",
                        help="输出数据目录")
    parser.add_argument("--samples_per_type", type=int, default=100,
                        help="每种类型生成的样本数量")
    # 添加fake参数
    parser.add_argument("--fake", action="store_true",
                        help="使用假搜索而不是真实搜索")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(args.config_path):
        args.config_path = os.path.join(base_dir, args.config_path.lstrip("./"))
    if not os.path.isabs(args.input_questions):
        args.input_questions = os.path.join(base_dir, args.input_questions.lstrip("./"))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir.lstrip("./"))
    
    # 初始化推理类并生成数据
    inference = DeepSeekInference(args.model_path, args.config_path, fake_search=args.fake)
    inference.generate_all_data(args.input_questions, args.output_dir, args.samples_per_type)

if __name__ == "__main__":
    main()