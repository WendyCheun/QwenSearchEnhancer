import os
import json
import yaml
from typing import List, Dict, Any, Union

def ensure_dir(directory: str) -> None:
    """确保目录存在，如果不存在则创建"""
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_jsonl(file_path: str) -> List[Any]:
    """加载JSONL文件"""
    if not os.path.exists(file_path):
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Any], file_path: str) -> None:
    """保存数据为JSONL文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_yaml(file_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """保存数据为YAML文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def merge_jsonl_files(input_files: List[str], output_file: str) -> None:
    """合并多个JSONL文件"""
    all_data = []
    for file_path in input_files:
        all_data.extend(load_jsonl(file_path))
    
    save_jsonl(all_data, output_file)