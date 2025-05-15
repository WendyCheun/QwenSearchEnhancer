import os
import json
import logging
import argparse
import sys
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch.nn.functional as F
from torch import nn

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    model_name: str = field(
        default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
        metadata={"help": "预训练模型的路径或名称，例如 'Qwen/Qwen2.5-0.5B-Instruct'"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用LoRA进行训练"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "需要应用LoRA的模块名称列表"}
    )


@dataclass
class DataArguments:
    """
    数据相关参数
    """
    train_file: str = field(
        default=None, metadata={"help": "训练数据文件路径"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "验证数据文件路径"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "输入序列的最大长度"}
    )


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    预处理函数，将对话数据转换为模型输入格式

    Args:
        examples: 数据样本
        tokenizer: 分词器
        max_seq_length: 最大序列长度

    Returns:
        处理后的数据
    """
    # 处理对话数据
    conversations = examples["conversations"]
    inputs = []
    search_labels = []

    for conv in conversations:
        # 提取对话内容
        formatted_text = ""
        for turn in conv["conversations"]:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"

        inputs.append(formatted_text)

        # 提取搜索标签
        if "metadata" in conv and "need_search" in conv["metadata"]:
            search_labels.append(1 if conv["metadata"]["need_search"] else 0)
        else:
            # 如果没有标签，使用-100表示忽略
            search_labels.append(-100)

    # 对输入文本进行编码
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 准备标签
    labels = model_inputs["input_ids"].clone()

    # 将填充token的标签设为-100，在计算损失时会被忽略
    labels[labels == tokenizer.pad_token_id] = -100

    # 添加标签到模型输入
    model_inputs["labels"] = labels

    # 添加搜索标签
    if any(label != -100 for label in search_labels):
        model_inputs["search_labels"] = torch.tensor(search_labels)

    return model_inputs


def create_model(model_name, use_lora=True, lora_config=None):
    """
    创建模型

    Args:
        model_name: 模型名称或路径
        use_lora: 是否使用LoRA
        lora_config: LoRA配置

    Returns:
        配置好的模型
    """
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        "/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info(f"Base model loaded. Hidden size: {model.config.hidden_size}")

    model.gradient_checkpointing_enable()
    # 如果使用LoRA，配置模型
    if use_lora:
        # 准备模型进行LoRA微调
        model = prepare_model_for_kbit_training(model)

        # 如果没有提供LoRA配置，使用默认配置
        if lora_config is None:
            # 为Qwen2.5模型设置默认的target_modules
            target_modules = ["q_proj", "k_proj", "v_proj",
                              "o_proj", "gate_proj", "up_proj", "down_proj"]

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

        # 应用LoRA配置到模型
        model = get_peft_model(model, lora_config)
        logger.info("LoRA configured.")
        model.print_trainable_parameters()
    # 添加搜索分类头
    hidden_size = model.config.hidden_size
    model.search_classifier = nn.Linear(hidden_size, 1).to(model.device)
    logger.info(f"Search classifier added with input size: {hidden_size}")

    original_forward = model.forward

    '''
    def forward_with_search(self, **kwargs):
        search_labels = kwargs.pop("search_labels", None)
        outputs = original_forward(**kwargs)
        if search_labels is not None and hasattr(self, "search_classifier"):
            # 获取最后一层隐藏状态
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden_state = outputs.hidden_states[-1]
            else:
                last_hidden_state = outputs.last_hidden_state if hasattr(
                    outputs, "last_hidden_state") else None
                if last_hidden_state is not None:
                    # 使用序列的第一个token的表示进行分类
                    first_token_hidden = last_hidden_state[:, 0, :]
                    search_logits = self.search_classifier(first_token_hidden)
                    # 计算搜索判断损失
                    search_loss = F.binary_cross_entropy_with_logits(
                        search_logits, search_labels.float().view(-1, 1))
                    # 添加到总损失
                    outputs.loss = search_loss
                    outputs.search_logits = search_logits
        return outputs
    '''

    def forward_with_search(self, **kwargs):
        search_labels = kwargs.pop("search_labels", None)
        outputs = original_forward(**kwargs)

        if search_labels is not None and hasattr(self, "search_classifier"):
            # 获取最后一层隐藏状态
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden_state = outputs.hidden_states[-1]
            else:
                last_hidden_state = outputs.last_hidden_state if hasattr(
                    outputs, "last_hidden_state") else None

            if last_hidden_state is not None:
                # 方法1：使用序列的平均表示进行分类（考虑整个对话）
                # 获取注意力掩码以排除填充token
                attention_mask = kwargs.get("attention_mask", None)
                if attention_mask is not None:
                    # 使用注意力掩码计算平均值（只考虑非填充token）
                    mask_expanded = attention_mask.unsqueeze(
                        -1).expand(last_hidden_state.size())
                    sum_hidden = torch.sum(
                        last_hidden_state * mask_expanded, dim=1)
                    seq_lengths = torch.sum(
                        attention_mask, dim=1, keepdim=True)
                    avg_hidden = sum_hidden / (seq_lengths + 1e-8)  # 避免除零
                    search_logits = self.search_classifier(avg_hidden)
                else:
                    # 如果没有注意力掩码，使用整个序列的平均值
                    avg_hidden = torch.mean(last_hidden_state, dim=1)
                    search_logits = self.search_classifier(avg_hidden)

                # 计算搜索判断损失
                search_loss = F.binary_cross_entropy_with_logits(
                    search_logits, search_labels.float().view(-1, 1))

                # 添加到总损失（如果已有损失，则组合）
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    # 使用加权组合
                    lm_loss_weight = 0.7  # 语言模型损失权重
                    search_loss_weight = 0.3  # 搜索分类损失权重
                    combined_loss = lm_loss_weight * outputs.loss + search_loss_weight * search_loss
                    outputs.loss = combined_loss
                else:
                    outputs.loss = search_loss

                outputs.search_logits = search_logits

        return outputs

    # 替换forward方法
    model.forward = forward_with_search.__get__(model, type(model))
    return model


def train_model(args):
    """使用完整参数进行训练"""
    # 解析命令行参数
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=args)

    # 设置随机种子
    set_seed(training_args.seed)

    # 创建LoRA配置
    if model_args.use_lora:
        if model_args.target_modules is None:
            # 为Qwen2.5模型设置默认的target_modules
            model_args.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        logger.info(f"使用LoRA训练，target_modules: {model_args.target_modules}")

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        lora_config = None

    # 创建模型
    model = create_model(
        model_args.model_name,
        use_lora=model_args.use_lora,
        lora_config=lora_config
    )

    # 加载tokenizer
    logger.info(f"从路径加载模型: {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    extension = data_args.train_file.split(
        ".")[-1] if data_args.train_file else "json"
    raw_datasets = load_dataset(extension, data_files=data_files)

    # 预处理数据
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(
            examples, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # 初始化标准Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if "train" in tokenized_datasets else None,
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        tokenizer=tokenizer,
    )

    # 开始训练
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 评估
    if training_args.do_eval:
        logger.info("*** 评估 ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main():
    """主函数，提供两种使用方式"""
    # 检查是否使用简化参数
    if any(arg.startswith('--simple_mode') for arg in sys.argv):  # 简化模式
        parser = argparse.ArgumentParser(
            description="启动Qwen2.5-0.5B-Instruct的训练")
        parser.add_argument(
            "--simple_mode", action="store_true", help="使用简化参数模式")
        parser.add_argument("--train_file", type=str,
                            default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_to_train/train.json", help="训练数据文件路径")
        parser.add_argument("--validation_file", type=str,
                            default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_to_val/val.json", help="验证数据文件路径")
        parser.add_argument("--output_dir", type=str,
                            default="/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_train_ok", help="输出目录")
        parser.add_argument("--model_name", type=str,
                            default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct", help="模型名称或路径")
        parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
        parser.add_argument("--gradient_accumulation_steps",
                            type=int, default=8, help="梯度累积步数")
        parser.add_argument("--learning_rate", type=float,
                            default=1e-3, help="学习率")
        parser.add_argument("--num_train_epochs", type=int,
                            default=5, help="训练轮数")
        parser.add_argument("--max_seq_length", type=int,
                            default=2048, help="最大序列长度")
        parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int,
                            default=16, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float,
                            default=0.05, help="LoRA dropout")
        parser.add_argument("--no_lora", action="store_true", help="不使用LoRA")
        parser.add_argument("--offline", action="store_true", help="使用离线模式")

        args = parser.parse_args()

        # 如果使用离线模式，设置环境变量
        if args.offline:
            transformers.utils.hub.HUGGINGFACE_OFFLINE = True
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # 构建完整参数列表
        full_args = [
            f"--model_name={args.model_name}",
            f"--train_file={args.train_file}",
            f"--output_dir={args.output_dir}",
            f"--per_device_train_batch_size={args.batch_size}",
            f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
            f"--learning_rate={args.learning_rate}",
            f"--num_train_epochs={args.num_train_epochs}",
            f"--max_seq_length={args.max_seq_length}",
            f"--lora_r={args.lora_r}",
            f"--lora_alpha={args.lora_alpha}",
            f"--lora_dropout={args.lora_dropout}",
            f"--save_strategy=epoch",
            f"--logging_steps=10",
            f"--fp16",
            f"--use_lora={'False' if args.no_lora else 'True'}",
            f"--overwrite_output_dir=false",
            f"--do_train=true",
            # f"--local_files_only" if args.offline else ""
        ]

        if args.validation_file:
            full_args.extend([
                f"--validation_file={args.validation_file}",
                # f"--evaluation_strategy=epoch",
                f"--do_eval"
            ])
        # 调用训练函数
        train_model(full_args)
    else:
        # 完整模式 (类似原来的trainer.py)
        train_model(None)  # 使用sys.argv


if __name__ == "__main__":
    main()
