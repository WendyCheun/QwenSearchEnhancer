import requests
import os
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class WebSearchEngine:
    """实现网络搜索的引擎类，可以是真实搜索或假搜索"""
    def __init__(self, api_key=None, use_fake_search=False, model_path=None):
        """
        初始化搜索引擎
        
        Args:
            api_key: 搜索API密钥
            use_fake_search: 是否使用假搜索
            model_path: 用于假搜索的模型路径
        """
        self.use_fake_search = use_fake_search
        self.model_path = model_path
        
        if not use_fake_search:
            # 真实搜索模式
            self.api_key = api_key or os.environ.get("SEARCHAPI_KEY")
            if not self.api_key:
                print("警告: 未设置SEARCHAPI_KEY环境变量，搜索功能可能受限")
        else:
            # 假搜索模式
            self.real_examples = self.learn_real_search()
            
            # 初始化模型（如果提供了模型路径）
            if model_path:
                try:
                    from vllm import LLM, SamplingParams
                    print(f"正在加载模型: {model_path}")
                    self.model = LLM(
                        model=model_path,
                        tensor_parallel_size=4,
                        gpu_memory_utilization=0.9,
                        trust_remote_code=True
                    )
                    self.sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=2048
                    )
                    print("模型加载完成")
                except ImportError:
                    print("警告: 无法导入vllm，将尝试使用OpenAI API")
                    self.model = None
            else:
                self.model = None
                
            # 如果没有本地模型，尝试使用OpenAI API
            if self.model is None:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(
                        api_key="EMPTY",
                        base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
                    )
                    try:
                        self.model_name = self.client.models.list().data[0].id
                        print(f"使用OpenAI API，模型: {self.model_name}")
                    except:
                        self.model_name = "gpt-3.5-turbo"
                        print(f"无法获取模型列表，使用默认模型: {self.model_name}")
                except ImportError:
                    print("警告: 无法导入openai，假搜索引擎将返回模拟数据")
                    self.client = None
    
    def learn_real_search(self) -> List[Dict[str, Any]]:
        """
        学习真实搜索结果，从现有数据中提取搜索关键词和结果
        
        Returns:
            包含搜索关键词和结果的示例列表
        """
        examples = []
        
        # 尝试从现有数据中加载示例
        data_paths = [
            "/remote-home1/wxzhang/QwenSearchEnhancer/data/synthetic/search_and_answer.jsonl",
            "/remote-home1/wxzhang/QwenSearchEnhancer/data/synthetic/all_synthetic_data.jsonl"
        ]
        
        for data_path in data_paths:
            if os.path.exists(data_path):
                print(f"从 {data_path} 加载真实搜索示例...")
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            if data.get("type") == "search_and_answer" and "metadata" in data:
                                metadata = data["metadata"]
                                if "search_keyword" in metadata and "search_results" in metadata:
                                    examples.append({
                                        "keyword": metadata["search_keyword"],
                                        "results": metadata["search_results"]
                                    })
                except Exception as e:
                    print(f"加载真实搜索示例出错: {e}")
        
        if not examples:
            print("未找到真实搜索示例，将使用默认示例")
            # 添加一些默认示例
            examples = [
                {
                    "keyword": "人工智能最新进展",
                    "results": [
                        "人工智能领域的最新进展包括大型语言模型的突破，如GPT-4和Claude等模型展现出更强的理解能力和生成能力。这些模型能够处理多模态输入，包括文本、图像甚至视频，并能生成高质量的内容。",
                        "在医疗领域，AI系统已经能够辅助诊断多种疾病，准确率在某些任务上超过了人类专家。例如，DeepMind的AlphaFold2在蛋白质结构预测方面取得了重大突破，这对药物研发具有革命性意义。",
                        "自动驾驶技术也在不断进步，特斯拉、Waymo等公司的自动驾驶系统正在多个城市进行测试。同时，AI在气候变暖研究、材料科学等领域也有重要应用，帮助科学家解决复杂问题。"
                    ]
                },
                {
                    "keyword": "量子计算机应用",
                    "results": [
                        "量子计算机在密码学领域有重要应用，理论上可以破解当前广泛使用的RSA加密算法。为应对这一挑战，研究人员正在开发抗量子密码学算法，以确保未来通信安全。",
                        "在药物研发方面，量子计算机可以模拟分子结构和相互作用，大幅加速新药发现过程。制药公司如默克和拜耳已与量子计算公司合作，探索这一应用。",
                        "金融领域也在探索量子计算应用，如投资组合优化、风险分析和欺诈检测。摩根大通和高盛等金融机构已建立专门团队研究量子计算技术。"
                    ]
                }
            ]
        
        print(f"成功加载 {len(examples)} 个真实搜索示例")
        return examples
    
    def _format_examples(self, num_examples: int = 2) -> str:
        """格式化真实搜索示例"""
        if not self.real_examples:
            return ""
        
        # 随机选择示例
        selected_examples = random.sample(self.real_examples, min(num_examples, len(self.real_examples)))
        
        formatted = ""
        for i, example in enumerate(selected_examples):
            formatted += f"关键词: {example['keyword']}\n"
            for j, result in enumerate(example['results'], 1):
                formatted += f"结果{j}: {result}\n"
            if i < len(selected_examples) - 1:
                formatted += "\n"
        
        return formatted
    
    def search(self, keyword: str, top_k: int = 3) -> List[str]:
        """
        执行搜索并返回结果列表
        
        Args:
            keyword: 搜索关键词
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表，每个结果为一段文本
        """
        if not self.use_fake_search:
            # 使用真实搜索API
            return self._real_search(keyword, top_k)
        else:
            # 使用假搜索
            return self._fake_search(keyword, top_k)
    
    def _real_search(self, keyword: str, top_k: int = 3) -> List[str]:
        """执行真实搜索"""
        try:
            # 使用SearchAPI.io进行搜索
            api_url = "https://www.searchapi.io/api/v1/search"
            
            # 设置语言和国家代码
            language = "zh-cn"
            country_code = language.split('-')[-1].upper() if '-' in language else "US"
            
            params = {
                "engine": "google",
                "q": keyword,
                "api_key": self.api_key,
                "gl": country_code,  # 国家代码
                "hl": language,      # 语言代码
                "num": top_k
            }
            
            headers = {
                "User-Agent": "QwenSearchEnhancer/1.0"
            }
            
            response = requests.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            # 从搜索结果中提取摘要
            if "organic_results" in data:
                for result in data["organic_results"][:top_k]:
                    if "snippet" in result:
                        # 限制每个结果最多500字
                        results.append(result["snippet"][:500])
            
            return results if results else [f"未找到关于 '{keyword}' 的搜索结果"]
            
        except Exception as e:
            print(f"搜索出错: {e}")
            return [f"搜索 '{keyword}' 时发生错误: {str(e)}"]
    
    def _fake_search(self, keyword: str, top_k: int = 3) -> List[str]:
        """执行假搜索"""
        # 准备示例
        examples = self._format_examples(num_examples=2)
        
        # 构建提示
        prompt = f"""我将扮演一个搜索引擎，生成关于"{keyword}"的搜索结果。请参考以下真实搜索结果的例子，生成{min(top_k, 10)}个类似的搜索结果。每个结果应该是一段不超过500字的摘要，内容要真实、客观、信息丰富。

真实搜索结果示例:
{examples}

请生成关于"{keyword}"的{min(top_k, 10)}个搜索结果:"""
        
        if hasattr(self, 'model') and self.model:
            # 使用本地模型
            response = self.model.generate(prompt, self.sampling_params)
            results_text = response[0].outputs[0].text.strip()
        elif hasattr(self, 'client') and self.client:
            # 使用OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7,
                    n=1
                )
                results_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"使用OpenAI API生成搜索结果出错: {e}")
                return self._generate_mock_results(keyword, top_k)
        else:
            # 无法使用模型，返回模拟数据
            return self._generate_mock_results(keyword, top_k)
        
        # 处理结果，尝试多种分割方式
        results = []
        
        # 尝试按"结果X:"格式分割
        import re
        pattern = r'结果\s*\d+\s*[:：]\s*(.*?)(?=结果\s*\d+\s*[:：]|$)'
        matches = re.findall(pattern, results_text, re.DOTALL)
        if matches and len(matches) >= top_k:
            results = [m.strip() for m in matches[:top_k]]
        
        # 如果上面的方法没有找到足够的结果，尝试按空行分割
        if len(results) < top_k:
            paragraphs = [p.strip() for p in results_text.split("\n\n") if p.strip()]
            clean_paragraphs = []
            for p in paragraphs:
                # 移除可能的编号前缀
                clean_p = re.sub(r'^\d+[\.\)、]\s*', '', p)
                clean_p = re.sub(r'^结果\s*\d+\s*[:：]\s*', '', clean_p)
                clean_paragraphs.append(clean_p)
            
            if clean_paragraphs:
                results = clean_paragraphs[:top_k]
        
        # 如果还是没有足够的结果，尝试按行分割
        if len(results) < top_k:
            lines = [line.strip() for line in results_text.split("\n") if line.strip()]
            clean_lines = []
            for line in lines:
                # 移除可能的编号前缀和结果标记
                if not line.startswith("关键词") and not line.lower().startswith("keyword"):
                    clean_line = re.sub(r'^\d+[\.\)、]\s*', '', line)
                    clean_line = re.sub(r'^结果\s*\d+\s*[:：]\s*', '', clean_line)
                    if len(clean_line) > 20:  # 只保留有实质内容的行
                        clean_lines.append(clean_line)
            
            if clean_lines:
                results = clean_lines[:top_k]
        
        # 如果所有方法都失败，返回一个简单的结果
        if not results:
            return self._generate_mock_results(keyword, top_k)
        
        return results[:top_k]
    
    def _generate_mock_results(self, keyword: str, top_k: int = 3) -> List[str]:
        """生成模拟搜索结果"""
        return [
            f"关于 '{keyword}' 的模拟搜索结果：{keyword}是一个重要的研究领域，近年来取得了显著进展。专家们认为，未来几年这一领域将继续快速发展，带来更多创新应用。",
            f"根据最新研究，{keyword}技术已经在多个行业得到应用，包括医疗、金融和教育等。这些应用显著提高了效率并降低了成本。",
            f"{keyword}相关的市场规模预计将在未来5年内增长300%，吸引了大量投资。领先企业正在加大研发投入，争夺市场份额。"
        ][:top_k]

class SearchTool:
    """搜索工具封装，用于在数据生成过程中模拟搜索功能"""
    def __init__(self, search_engine=None, use_fake_search=False, model_path=None):
        """
        初始化搜索工具
        
        Args:
            search_engine: 搜索引擎实例，如果为None则创建新实例
            use_fake_search: 是否使用假搜索引擎
            model_path: 假搜索引擎的模型路径
        """
        if search_engine:
            self.search_engine = search_engine
        else:
            self.search_engine = WebSearchEngine(use_fake_search=use_fake_search, model_path=model_path)
        
    def search(self, keyword: str, top_k: int = 3) -> List[str]:
        """执行搜索并返回结果"""
        try:
            results = self.search_engine.search(keyword, top_k)
            return results
        except Exception as e:
            print(f"搜索出错: {e}")
            return [f"搜索 '{keyword}' 时发生错误"]
    
    def format_search_results(self, results: List[str]) -> str:
        """将搜索结果格式化为字符串"""
        formatted = "搜索结果:\n"
        for i, result in enumerate(results, 1):
            formatted += f"[{i}] {result}\n\n"
        return formatted.strip()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="执行搜索")
    parser.add_argument("--keyword", type=str, default="人工智能最新进展",
                        help="搜索关键词")
    parser.add_argument("--top_k", type=int, default=3,
                        help="返回结果数量")
    parser.add_argument("--fake", action="store_true",
                        help="使用假搜索引擎")
    parser.add_argument("--model_path", type=str, default=None,
                        help="假搜索引擎的模型路径")
    
    args = parser.parse_args()
    
    search_tool = SearchTool(use_fake_search=args.fake, model_path=args.model_path)
    results = search_tool.search(args.keyword, args.top_k)
    print(search_tool.format_search_results(results))

