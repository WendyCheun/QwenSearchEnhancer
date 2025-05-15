import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams

class QuestionDirectGenerator:
    """直接生成高质量问题的类，不依赖于从种子数据中提取关键词"""
    
    def __init__(self, model_path: str, categories: List[str] = None):
        """
        初始化问题生成器
        
        Args:
            model_path: 用于生成问题的模型路径
            categories: 问题类别列表，如果为None则使用默认类别
        """
        self.model_path = model_path
        
        # 设置默认问题类别
        self.categories = categories or [
            # 原有类别
            "科技与创新", "经济与金融", "环境与可持续发展", "健康与医疗", 
            "教育与学习", "社会与文化", "政治与国际关系", "哲学与伦理",
            "艺术与文学", "历史与考古", "数学与逻辑", "物理与宇宙",
            "生物与生命科学", "心理学与行为", "工程与建筑", "法律与法规",
            
            # 扩充的科技相关类别
            "人工智能与机器学习", "大数据与云计算", "区块链与加密货币", "量子计算", 
            "虚拟现实与增强现实", "物联网技术", "网络安全", "5G与通信技术",
            "机器人技术", "自动驾驶", "航空航天", "半导体与芯片",
            
            # 扩充的商业与经济类别
            "创业与商业模式", "市场营销与品牌", "电子商务", "供应链管理",
            "投资与理财", "风险管理", "国际贸易", "宏观经济政策",
            "数字经济", "共享经济", "劳动力市场", "企业管理",
            
            # 扩充的社会科学类别
            "社会学研究", "人类学", "考古学新发现", "城市规划与发展",
            "人口统计学", "移民与全球化", "性别研究", "媒体与传播",
            "公共政策", "社会福利", "犯罪学", "教育政策",
            
            # 扩充的自然科学类别
            "天文学与宇宙探索", "地质学与地球科学", "气象学与气候", "海洋学",
            "粒子物理", "材料科学", "纳米技术", "能源技术",
            "生态学", "进化生物学", "遗传学", "神经科学",
            
            # 扩充的医疗与健康类别
            "公共卫生", "流行病学", "营养学", "精准医疗",
            "心理健康", "老年医学", "儿童健康", "慢性疾病管理",
            "医疗技术", "生物医学工程", "药物研发", "替代医学",
            
            # 扩充的文化与艺术类别
            "电影与影视", "音乐与表演艺术", "文学创作", "视觉艺术",
            "建筑设计", "时尚与设计", "游戏设计", "数字艺术",
            "文化遗产保护", "博物馆学", "传统文化", "跨文化交流",
            
            # 扩充的生活与日常类别
            "饮食与烹饪", "旅游与探险", "家居与装修", "个人成长",
            "亲子关系", "宠物护理", "园艺与植物", "运动与健身",
            "时间管理", "工作与生活平衡", "消费者权益", "可持续生活方式",
            
            # 扩充的前沿交叉领域
            "生物信息学", "计算社会科学", "环境心理学", "行为经济学",
            "认知科学", "数字人文", "生物伦理学", "科技伦理",
            "可持续设计", "循环经济", "智慧城市", "未来学研究"
        ]
        
        # 初始化模型
        print(f"正在加载模型: {model_path}")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=0.9,  # 使用较高的温度以增加多样性
            top_p=0.95,
            max_tokens=512
        )
        print("模型加载完成")
    
    def generate_questions(self, num_questions: int, output_path: str, 
                          questions_per_category: int = None) -> List[str]:
        """
        直接生成高质量问题
        
        Args:
            num_questions: 要生成的问题总数量
            output_path: 输出文件路径
            questions_per_category: 每个类别生成的问题数量，如果为None则平均分配
            
        Returns:
            生成的问题列表
        """
        # 计算每个类别需要生成的问题数量
        if questions_per_category is None:
            questions_per_category = num_questions // len(self.categories)
            # 处理余数
            remainder = num_questions % len(self.categories)
        else:
            # 确保总数不超过要求
            if questions_per_category * len(self.categories) > num_questions:
                questions_per_category = num_questions // len(self.categories)
            remainder = num_questions - questions_per_category * len(self.categories)
        
        print(f"将为每个类别生成 {questions_per_category} 个问题，共 {len(self.categories)} 个类别")
        if remainder > 0:
            print(f"剩余的 {remainder} 个问题将分配给前 {remainder} 个类别")
        
        generated_questions = []
        
        # 为每个类别生成问题
        for i, category in enumerate(tqdm(self.categories, desc="处理类别")):
            # 确定当前类别需要生成的问题数量
            current_category_count = questions_per_category + (1 if i < remainder else 0)
            
            if current_category_count <= 0:
                continue
                
            # 构建提示
            prompt = self._build_prompt_for_category(category)
            
            # 批量生成该类别的问题
            batch_questions = self._generate_batch_questions(prompt, current_category_count)
            
            # 添加到结果列表
            generated_questions.extend(batch_questions)
            
            print(f"已为类别 '{category}' 生成 {len(batch_questions)} 个问题")
        
        # 保存生成的问题
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for question in generated_questions:
                    f.write(json.dumps({"question": question}, ensure_ascii=False) + "\n")
            print(f"已将 {len(generated_questions)} 个生成的问题保存至 {output_path}")
        
        return generated_questions
    
    def _build_prompt_for_category(self, category: str) -> str:
        """为特定类别构建提示"""
        return f"""请直接生成5个关于"{category}"的高质量问题。这些问题应该：
1. 具有开放性，需要深入思考和搜索才能回答
2. 涉及该领域的前沿话题或重要概念
3. 能够引发深度讨论和思考
4. 避免简单的是非题或单一答案的问题
5. 问题应该清晰、具体且有深度
6. 每个问题应该是独立的，不要有序号或编号
7. 直接输出问题文本，不要有思考过程或解释

重要提示：
- 直接列出问题，每行一个问题
- 不要包含任何思考过程
- 不要使用编号
- 不要包含"</think>"等标记
- 确保问题多样化，避免相似的问题
- 确保问题是中文的

请直接列出这些问题："""
    
    def _generate_batch_questions(self, prompt: str, count: int) -> List[str]:
        """批量生成问题"""
        questions = []
        attempts = 0
        max_attempts = 5  # 最大尝试次数
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            # 构建更明确的提示，指导模型直接输出问题，避免思考过程
            current_prompt = f"{prompt}\n\n请直接列出{min(count - len(questions), 5)}个问题，每个问题一行，不要有编号，不要有思考过程，不要有'</think>'等标记。"
            
            # 生成问题
            response = self.model.generate(current_prompt, self.sampling_params)
            raw_text = response[0].outputs[0].text.strip()
            
            # 使用更智能的方式分割问题
            candidate_questions = self._extract_questions(raw_text)
            
            # 过滤和清理问题
            for question in candidate_questions:
                # 清理问题格式
                clean_q = self._clean_question(question)
                
                # 质量检查
                if clean_q and self._check_question_quality(clean_q) and clean_q not in questions:
                    questions.append(clean_q)
                    
                # 如果已经收集到足够的问题，就停止
                if len(questions) >= count:
                    break
        
        # 如果尝试多次后仍然没有足够的问题，记录警告
        if len(questions) < count:
            print(f"警告：尝试 {attempts} 次后仅生成了 {len(questions)} 个问题，少于请求的 {count} 个")
        
        return questions[:count]  # 确保不超过请求的数量
    
    def _extract_questions(self, text: str) -> List[str]:
        """
        智能提取文本中的问题
        使用更复杂的逻辑来识别完整的问题
        """
        # 首先按行分割
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 过滤掉明显不是问题的行
        questions = []
        for line in lines:
            # 跳过思考过程、元描述或指令
            if any(skip in line.lower() for skip in ['</think>', '好的', '首先', '接下来', '然后', '最后', '例如', '分析', '第']):
                continue
                
            # 跳过太短的行
            if len(line) < 10:
                continue
                
            # 跳过不以问号结尾且不包含问号的行
            if not ('?' in line or '？' in line):
                continue
                
            questions.append(line)
        
        return questions
    
    def _check_question_quality(self, question: str) -> bool:
        """
        检查问题质量
        返回True表示问题质量合格，False表示不合格
        """
        # 检查问题长度
        if len(question) < 15 or len(question) > 200:
            return False
            
        # 检查是否包含模型思考的标记
        thinking_markers = ['</think>', '好的', '首先', '接下来', '然后', '最后', '例如', '我需要', '我要', '我会']
        if any(marker in question for marker in thinking_markers):
            return False
            
        # 检查是否是指令而非问题
        instruction_markers = ['请生成', '请列出', '生成问题', '列出问题']
        if any(marker in question for marker in instruction_markers):
            return False
            
        # 检查是否包含编号
        if any(pattern in question for pattern in ['1.', '2.', '问题1', '第1']):
            return False
            
        # 检查是否是元问题（关于问题本身的问题）
        meta_markers = ['这个问题', '一个好的问题', '为什么这是']
        if any(marker in question for marker in meta_markers):
            return False
            
        # 检查是否包含英文（可选，取决于需求）
        # if re.search(r'[a-zA-Z]{5,}', question):
        #     return False
            
        return True
    
    def _clean_question(self, text: str) -> str:
        """清理和格式化问题文本"""
        # 移除可能的序号前缀
        text = text.strip()
        
        # 移除数字编号
        if text and text[0].isdigit() and '. ' in text[:10]:
            text = text.split('. ', 1)[1]
        
        # 移除其他可能的前缀
        prefixes = ["问题：", "问题:", "Q:", "Q：", "问：", "问:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # 移除思考过程标记
        thinking_markers = ["</think>", "<think>"]
        for marker in thinking_markers:
            if marker in text:
                text = text.replace(marker, "").strip()
        
        # 确保问题以问号结尾
        if text and not text.endswith("?") and not text.endswith("？"):
            text += "？"
        
        # 移除多余的问号
        while text.endswith("？？") or text.endswith("??"):
            text = text[:-1]
        
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="直接生成高质量问题")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="模型路径")
    parser.add_argument("--num_questions", type=int, default=500,
                        help="要生成的问题总数量")
    parser.add_argument("--output_path", type=str, default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/direct_generated_questions.jsonl",
                        help="输出文件路径")
    parser.add_argument("--categories_file", type=str, default="",
                        help="问题类别文件路径，每行一个类别，如果不提供则使用默认类别")
    
    args = parser.parse_args()
    
    # 加载自定义类别（如果提供）
    categories = None
    if args.categories_file and os.path.exists(args.categories_file):
        with open(args.categories_file, 'r', encoding='utf-8') as f:
            categories = [line.strip() for line in f if line.strip()]
        print(f"从 {args.categories_file} 加载了 {len(categories)} 个自定义类别")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 生成问题
    generator = QuestionDirectGenerator(args.model_path, categories)
    generator.generate_questions(args.num_questions, args.output_path)