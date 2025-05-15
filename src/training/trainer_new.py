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

# ====== 新增导入 ======
import numpy as np
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import EvalPrediction
# ======================

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
    （此函数无需修改，因为它已经正确地从conv['metadata']中提取search_labels）

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
        # 根据你的新数据格式（metadata在conversation列表外面）
        # 你需要在 formatted_text 的开头加入与 metadata 对应的 token 或标记
        # 例如，如果你在数据预处理时决定添加 [<search_needed>] 或 [<no_search>]
        # 那么formatted_text应该以这个标记开头
        # *** 注意：这里的代码假设你的 formatted_text 已经包含了开头的标记，
        # *** 否则你需要修改这里以实际添加标记 token。
        # *** 例如：formatted_text = f"<search_needed>\n" + formatted_text # 仅示例

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

        # 提取搜索标签 (这部分逻辑是正确的，与新数据格式匹配)
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

    # 准备语言模型标签 (这部分与搜索任务的标签是独立的)
    labels = model_inputs["input_ids"].clone()
    # 将填充token的标签设为-100，在计算LM损失时会被忽略
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    # 添加搜索标签到模型输入
    # 只有当批次中至少有一个有效的搜索标签时才添加 'search_labels' key
    if any(label != -100 for label in search_labels):
        model_inputs["search_labels"] = torch.tensor(search_labels)
    else:
        # 如果整个批次都没有有效的搜索标签，不添加 'search_labels' key
        # 这可以避免在 evaluation loop 中，当 batch 里全是 padding 或无标签数据时出错
        pass  # model_inputs 不会有 "search_labels" 这个key

    return model_inputs

# ====== 新增 compute_search_metrics 函数 ======


def compute_search_metrics(eval_pred: EvalPrediction):
    """
    计算搜索判断任务的准确率。

    Args:
        eval_pred: Trainer在评估时提供的 EvalPrediction 对象。
                   eval_pred.predictions 将是模型 forward 返回的 search_logits。
                   eval_pred.label_ids 将是 preprocess_function 添加的 search_labels。

    Returns:
        包含搜索准确率的字典。
    """
    search_logits = eval_pred.predictions
    search_labels = eval_pred.label_ids

    # 确保 logits 和 labels 的形状匹配，特别是 logits 如果是 (batch_size, 1) 需要 squeeze
    search_logits = search_logits.squeeze(-1)  # 形状变为 (batch_size,)

    # 过滤掉标签为 -100 的样本，这些样本不参与准确率计算
    valid_indices = search_labels != -100
    valid_search_labels = search_labels[valid_indices]
    valid_search_logits = search_logits[valid_indices]  # 只保留对应有效标签的 logits

    if len(valid_search_labels) == 0:
        # 如果当前评估批次没有有效的搜索标签，则无法计算准确率
        logger.warning(
            "No valid search labels found in the evaluation batch for metrics calculation.")
        return {"search_accuracy": 0.0}  # 或者返回空字典或N/A

    # 将 logits 转换为预测结果 (0 或 1)
    # 应用 Sigmoid 函数将 logits 转换为概率，然后使用阈值 0.5
    # 使用 np.exp 因为 logits 是 numpy array
    search_probabilities = 1 / (1 + np.exp(-valid_search_logits))
    search_predictions = (search_probabilities > 0.5).astype(int)

    # 计算准确率
    accuracy = accuracy_score(valid_search_labels, search_predictions)

    # 还可以选择在这里计算并报告 search_loss，但 Trainer 会报告总的 eval_loss
    # 并且 compute_metrics 接收的是 numpy 数组，计算 torch loss 会比较麻烦
    # 通常只在这里计算 accuracy 等指标

    return {"search_accuracy": float(accuracy)}  # 返回float类型
# ============================================


def create_model(model_name, use_lora=True, lora_config=None):
    """
    创建模型，修改forward方法以使用第一个token的隐藏状态进行搜索分类

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

    # 需要outputs包含last_hidden_state，确保配置中 output_hidden_states=True
    # 或者模型默认就会输出
    # model.config.output_hidden_states = True # 通常不是必须的，大部分LM模型最后一层隐藏状态是outputs.last_hidden_state

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
    # 将分类器添加到模型的设备上
    model.search_classifier = nn.Linear(hidden_size, 1).to(model.device)
    logger.info(f"Search classifier added with input size: {hidden_size}")

    # 保存原始的 forward 方法
    original_forward = model.forward

    # ====== 修改 forward 方法 ======
    def forward_with_search(self, input_ids=None, attention_mask=None, labels=None, search_labels=None, **kwargs):
        # 调用原始 forward 方法，传递所有相关参数
        # 原始 forward 会计算 LM Loss 并可能产生 outputs.logits (LM logits) 和 outputs.last_hidden_state
        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # 将LM labels传给原始forward计算LM loss
            **kwargs  # 传递其他可能的参数
        )

        search_loss = None
        search_logits = None

        # 仅当提供了 search_labels 且模型有 search_classifier 时才进行搜索分类计算
        # 检查 search_labels 中是否有非 -100 的标签
        if search_labels is not None and (search_labels != -100).any() and hasattr(self, "search_classifier"):
            # 获取最后一层隐藏状态 (对于大多数Transformers模型，这是可用的)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                # shape: (batch_size, sequence_length, hidden_size)
                last_hidden_state = outputs.last_hidden_state

                # *** 修改：使用序列的第一个token的表示进行分类 ***
                # 假设第一个 token 是你添加的指示 token 或用于分类的 token
                # shape of first_token_hidden: (batch_size, hidden_size)
                first_token_hidden = last_hidden_state[:, 0, :]

                # 通过搜索分类头获取 logits
                search_logits = self.search_classifier(
                    first_token_hidden)  # shape: (batch_size, 1)

                # 计算搜索判断损失
                # 只使用有效的 search_labels 进行损失计算 (search_labels != -100)
                valid_indices = search_labels != -100
                # shape: (num_valid_labels,)
                valid_search_labels = search_labels[valid_indices]
                # 需要对 search_logits 进行索引以匹配 valid_search_labels，并squeeze掉维度为1的尾部维度
                # search_logits 的 shape 是 (batch_size, 1)，valid_indices 的 shape 是 (batch_size,)
                # search_logits[valid_indices] 会得到 shape (num_valid_labels, 1)
                # 因此需要 squeeze(-1) 得到 shape (num_valid_labels,)
                valid_search_logits = search_logits[valid_indices].squeeze(-1)

                # 使用 BCEWithLogitsLoss 计算损失
                search_loss = F.binary_cross_entropy_with_logits(
                    valid_search_logits, valid_search_labels.float()
                )

                # 将搜索损失添加到总损失中
                # outputs.loss 是原始 LM Loss
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    lm_loss_weight = 0.7  # 语言模型损失权重
                    search_loss_weight = 0.3  # 搜索分类损失权重
                    # 只有在 search_loss 成功计算时才进行组合
                    if search_loss is not None:
                        outputs.loss = lm_loss_weight * outputs.loss + search_loss_weight * search_loss
                elif search_loss is not None:  # 原始outputs没有LM loss，但search_loss存在（不常见，除非原始forward返回None）
                    outputs.loss = search_loss

                # 将 search_logits 添加到 outputs 对象，方便 compute_metrics 访问
                # 保存完整的 batch logits，compute_metrics 会自己处理 -100
                outputs.search_logits = search_logits

            else:
                logger.warning(
                    "Model output does not have 'last_hidden_state'. Cannot perform search classification.")
                # 如果没有 last_hidden_state，则无法进行搜索分类，search_loss 和 search_logits 会保持 None

        # Trainer 的 compute_metrics 期望 forward 返回一个元组，第二个元素作为 predictions
        # 我们在有搜索任务时，返回 (总损失, search_logits)
        # 在没有搜索任务时 (比如纯LM预测模式，或没有search_labels的batch)，返回原始 outputs 对象
        # 只有当 search_logits 成功计算时才返回元组
        if search_logits is not None:
            # 返回格式 (loss, predictions) 供 Trainer 的 compute_metrics 使用
            return (outputs.loss, search_logits)
        else:
            # 如果当前 batch 没有 search_labels 或无法计算 search_logits，返回原始 outputs
            # 这确保了 Trainer 在没有搜索任务时也能正常工作
            return outputs
    # ===================================

    # 替换 forward 方法
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
        # ====== 添加 compute_metrics 函数 ======
        compute_metrics=compute_search_metrics if "validation" in tokenized_datasets else None,
        # ======================================
    )

    # 开始训练
    if training_args.do_train:
        logger.info("*** 开始训练 ***")
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 评估 (现在会自动计算 search_accuracy)
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
            f"--save_strategy=epoch",  # 可以结合 evaluation_strategy
            f"--logging_steps=10",
            f"--fp16",
            f"--use_lora={'False' if args.no_lora else 'True'}",
            f"--overwrite_output_dir=false",
            f"--do_train=true",
            # f"--local_files_only" if args.offline else "" # 这行可能需要根据实际离线环境调整
        ]

        if args.validation_file:
            full_args.extend([
                f"--validation_file={args.validation_file}",
                # f"--evaluation_strategy=epoch",  # 或者 "steps", 使Trainer定期运行评估
                f"--do_eval"
            ])
        # 调用训练函数
        train_model(full_args)
    else:
        # 完整模式 (类似原来的trainer.py)
        train_model(None)  # 使用sys.argv


if __name__ == "__main__":
    main()
