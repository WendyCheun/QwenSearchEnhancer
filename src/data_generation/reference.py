    def generate_search_and_answer(self, questions: List[str], output_path: str, 
                                  search_decisions: List[Dict[str, Any]] = None,
                                  complex_thinking_results: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """生成搜索并回答的数据
        
        Args:
            questions: 问题列表
            output_path: 输出路径
            search_decisions: 搜索决策结果，包含need_search字段
            complex_thinking_results: 复杂思考结果，当需要搜索时使用
        """
        results = []
        
        # 创建问题到决策和复杂思考的映射
        search_decision_map = {}
        complex_thinking_map = {}
        
        if search_decisions:
            for decision in search_decisions:
                search_decision_map[decision["instruction"]] = decision
                
        if complex_thinking_results:
            for thinking in complex_thinking_results:
                complex_thinking_map[thinking["instruction"]] = thinking
        
        for question in tqdm(questions, desc="生成搜索并回答数据"):
            # 检查是否需要搜索
            need_search = True  # 默认需要搜索
            if question in search_decision_map:
                need_search = search_decision_map[question]["metadata"]["need_search"]
            
            if need_search:
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
                
                # 第三步：使用复杂思考的结果作为搜索后的回答
                if question in complex_thinking_map:
                    final_answer = complex_thinking_map[question]["output"]
                else:
                    # 如果没有对应的复杂思考结果，则让模型基于搜索结果回答
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
                        "need_search": True,
                        "search_keyword": search_keyword,
                        "search_results": search_results,
                        "final_answer": final_answer
                    },
                    "type": "search_and_answer"
                }
            else:
                # 不需要搜索，直接让模型回答
                answer_prompt = f"{self.system_prompt}\n\n用户问题: {question}\n\n请直接回答用户问题:"
                answer_response = self.model.generate(answer_prompt, self.sampling_params)
                final_answer = answer_response[0].outputs[0].text.strip()
                
                full_output = f"思考：这个问题不需要最新信息，我可以直接回答。\n\n{final_answer}"
                
                result = {
                    "instruction": question,
                    "input": "",
                    "output": full_output,
                    "metadata": {
                        "need_search": False,
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
        
        # 为每种类型使用相同的问题集，保持顺序一致
        random.seed(42)  # 设置随机种子以确保可重复性
        
        # 从所有问题中随机选择sample_per_type个问题，但保持三个数据集使用相同的问题
        selected_questions = random.sample(questions, min(sample_per_type, len(questions)))
        
        # 生成各类数据
        results = {}
        
        # 1. 生成复杂思考数据
        complex_output_path = os.path.join(output_dir, "complex_thinking.jsonl")
        results["complex_thinking"] = self.generate_complex_thinking(selected_questions, complex_output_path)
        
        # 2. 生成搜索判断数据
        search_decision_output_path = os.path.join(output_dir, "search_decision.jsonl")
        results["search_decision"] = self.generate_search_decision(selected_questions, search_decision_output_path)
        
        # 3. 生成搜索并回答数据，使用前两个任务的结果
        search_answer_output_path = os.path.join(output_dir, "search_and_answer.jsonl")
        results["search_and_answer"] = self.generate_search_and_answer(
            selected_questions, 
            search_answer_output_path,
            search_decisions=results["search_decision"],
            complex_thinking_results=results["complex_thinking"]
        )
        
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