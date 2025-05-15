import os
import json
import argparse
from tqdm import tqdm
import random
from typing import Dict, List, Any, Tuple, Optional

def process_raw_data(data_path: str) -> List[Dict[str, Any]]:
    """
    处理原始数据文件，支持多种格式
    
    Args:
        data_path: 数据文件路径，可以是JSON或JSONL
        
    Returns:
        处理后的数据列表
    """
    all_data = []
    
    if data_path.endswith(".jsonl"):
        # 处理JSONL文件
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    else:
        # 处理JSON文件
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    return all_data

def convert_to_conversation_format(item: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """
    将单个数据项转换为统一的对话格式
    
    Args:
        item: 单个数据项
        
    Returns:
        转换后的对话格式数据，如果无法转换则返回None
    """
    result = None
    
    if "question" in item and "answer" in item:
        # 处理question/answer格式
        result = {
            "conversations": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        }
    elif "instruction" in item and "output" in item:
        # 处理instruction/output格式
        result = {
            "conversations": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
    elif "conversations" in item:
        # 如果数据已经是对话格式
        result = item
    elif "messages" in item:
        # 处理messages格式（OpenAI格式）
        conv = {"conversations": []}
        for msg in item["messages"]:
            if "role" in msg and "content" in msg:
                conv["conversations"].append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        if conv["conversations"]:
            result = conv
    
    # 如果成功转换了数据，并且原始数据中有metadata字段包含need_search标签，则保留该标签
    if result and "metadata" in item and "need_search" in item["metadata"]:
        result["metadata"] = {"need_search": item["metadata"]["need_search"]}
    
    return result

def format_qwen_conversation(conversation: List[Dict[str, str]]) -> str:
    """
    将对话数据格式化为Qwen2.5模型的对话模板
    
    Args:
        conversation: 对话数据列表
        
    Returns:
        格式化后的对话文本
    """
    formatted_text = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        
        if role == "user":
            formatted_text += "f"
def load_and_process_data(data_path: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    加载并处理数据，将所有数据转换为统一的对话格式
    
    Args:
        data_path: 数据路径，可以是文件或目录
        
    Returns:
        处理后的对话格式数据列表
    """
    all_data = []
    
    # 处理目录或单个文件
    if os.path.isdir(data_path):
        # 处理目录中的所有文件
        for filename in tqdm(os.listdir(data_path), desc="读取数据文件"):
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                file_path = os.path.join(data_path, filename)
                try:
                    data = process_raw_data(file_path)
                    all_data.extend(data)
                    print(f"从 {file_path} 读取了 {len(data)} 条数据")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    else:
        # 处理单个文件
        try:
            all_data = process_raw_data(data_path)
            print(f"从 {data_path} 读取了 {len(all_data)} 条数据")
        except Exception as e:
            print(f"处理文件 {data_path} 时出错: {e}")
    
    # 转换为对话格式
    conversations = []
    for item in tqdm(all_data, desc="转换数据格式"):
        conversation = convert_to_conversation_format(item)
        if conversation:
            conversations.append(conversation)
        else:
            print(f"警告: 无法识别的数据格式: {item.keys() if isinstance(item, dict) else type(item)}")
    
    print(f"共转换 {len(conversations)} 条对话")
    return conversations

def split_and_save_data(
    conversations: List[Dict[str, List[Dict[str, str]]]],
    output_path: str,
    split_ratio: float = 0.2
) -> Tuple[str, str]:
    """
    将对话数据划分为训练集和验证集并保存
    
    Args:
        conversations: 对话格式数据列表
        output_path: 输出目录
        split_ratio: 验证集比例
        
    Returns:
        训练集和验证集的文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 随机打乱数据
    random.seed(42)  # 设置随机种子以确保可重复性
    random.shuffle(conversations)
    
    # 划分训练集和验证集
    if split_ratio > 0:
        split_idx = int(len(conversations) * split_ratio)
        val_data = conversations[:split_idx]
        train_data = conversations[split_idx:]
    else:
        # 如果split_ratio为0，则所有数据都作为训练集
        train_data = conversations
        val_data = []
    
    # 保存数据
    train_path = os.path.join(output_path, "train.json")
    val_path = os.path.join(output_path, "validation.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump({"conversations": train_data}, f, ensure_ascii=False, indent=2)
    
    print(f"训练集保存至: {train_path}, 共 {len(train_data)} 条")
    
    if val_data:
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump({"conversations": val_data}, f, ensure_ascii=False, indent=2)
        print(f"验证集保存至: {val_path}, 共 {len(val_data)} 条")
    else:
        print("验证集比例为0，未生成验证集")
    
    return train_path, val_path

def prepare_conversation_data(data_path: str, output_path: str, split_ratio: float = 0.2) -> Tuple[str, str]:
    """
    完整的数据准备流程：加载、处理、转换、划分和保存
    
    Args:
        data_path: 数据路径，可以是文件或目录
        output_path: 输出目录
        split_ratio: 验证集比例
        
    Returns:
        训练集和验证集的文件路径
    """
    # 加载并处理数据
    conversations = load_and_process_data(data_path)
    
    # 划分并保存数据
    return split_and_save_data(conversations, output_path, split_ratio)

def process_qa_dataset_train(input_dir: str, output_dir: str, split_ratio: float = 0.2) -> None:
    """
    专门处理qa_dataset_train下的四个数据集
    
    Args:
        input_dir: qa_dataset_train目录路径
        output_dir: 输出目录路径
        split_ratio: 验证集比例
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取qa_dataset_train下的所有jsonl文件
    data_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            data_files.append(os.path.join(input_dir, filename))
    
    print(f"找到以下数据文件: {data_files}")
    
    # 处理每个数据文件
    for data_file in data_files:
        # 获取数据集名称（文件名，不含扩展名）
        dataset_name = os.path.splitext(os.path.basename(data_file))[0]
        print(f"\n开始处理数据集: {dataset_name}")
        
        # 为每个数据集创建单独的输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 加载数据
        try:
            all_data = process_raw_data(data_file)
            print(f"从 {data_file} 读取了 {len(all_data)} 条数据")
            
            # 转换为对话格式
            conversations = []
            for item in tqdm(all_data, desc="转换数据格式"):
                conversation = convert_to_conversation_format(item)
                if conversation:
                    conversations.append(conversation)
                else:
                    print(f"警告: 无法识别的数据格式: {item.keys() if isinstance(item, dict) else type(item)}")
            
            print(f"共转换 {len(conversations)} 条对话")
            
            # 随机打乱数据
            random.seed(42)  # 设置随机种子以确保可重复性
            random.shuffle(conversations)
            
            # 划分训练集和验证集
            split_idx = int(len(conversations) * split_ratio)
            val_data = conversations[:split_idx]
            train_data = conversations[split_idx:]
            
            # 保存数据
            train_path = os.path.join(dataset_output_dir, "train.json")
            val_path = os.path.join(dataset_output_dir, "validation.json")
            
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump({"conversations": train_data}, f, ensure_ascii=False, indent=2)
            
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump({"conversations": val_data}, f, ensure_ascii=False, indent=2)
            
            print(f"训练集保存至: {train_path}, 共 {len(train_data)} 条")
            print(f"验证集保存至: {val_path}, 共 {len(val_data)} 条")
            
            # 输出训练命令示例
            # print(f"训练命令示例:")
            # print(f"python /remote-home1/wxzhang/QwenSearchEnhancer/src/training/run_training.py --train_file {train_path} --validation_file {val_path} --output_dir /remote-home1/wxzhang/QwenSearchEnhancer/models/{dataset_name} --batch_size 4")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")

def main():
    """主函数，解析命令行参数并执行数据准备流程"""
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument("--data_path", type=str, default = "/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_val" , help="合成数据的路径")
    parser.add_argument("--output_path", type=str,  default = "/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_to_val", help="输出目录")
    parser.add_argument("--split_ratio", type=float, default=0, help="验证集比例")
    parser.add_argument("--process_qa_dataset", action="store_true", help="是否处理qa_dataset_train下的四个数据集")
    
    args = parser.parse_args()
    
    if args.process_qa_dataset:
        # 处理qa_dataset_train下的四个数据集
        print(f"处理qa_dataset_train下的数据集: {args.data_path}")
        process_qa_dataset_train(args.data_path, args.output_path, args.split_ratio)
        print("\n所有数据集处理完成!")
    else:
        # 处理单个数据集
        train_path, val_path = prepare_conversation_data(
            args.data_path, 
            args.output_path,
            args.split_ratio
        )
        
        print("数据准备完成！")
        # print(f"训练命令示例:")
        # print(f"python /remote-home1/wxzhang/QwenSearchEnhancer/src/training/run_training.py --train_file {train_path} --validation_file {val_path} --output_dir /path/to/output --batch_size 4")

if __name__ == "__main__":
    main()