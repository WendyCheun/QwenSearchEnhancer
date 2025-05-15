import json

input_file_path = "/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_to_train/train.json"
output_file_path = "/remote-home1/wxzhang/QwenSearchEnhancer/data/generated/qa_dataset_to_train/train1.json"


# def move_inner_metadata(data):
#
#     if isinstance(data, dict) and "conversations" in data and isinstance(data["conversations"], list):
#         for item in data["conversations"]:
#             if isinstance(item, dict) and "conversations" in item and isinstance(item["conversations"], list):
#                 conversations_list = item["conversations"]
#                 if conversations_list and isinstance(conversations_list[-1], dict) and "metadata" in conversations_list[-1]:
#                     metadata = conversations_list[-1].pop("metadata")
#                     conversations_list.insert(0, {"metadata": metadata})
#     return data
def move_metadata(data):
    updated_conversations = []
    for item in data["conversations"]:
        if "conversations" in item and item["conversations"]:
            metadata = item.pop("metadata")
            updated_item = {"metadata": metadata,
                            "conversations": item["conversations"]}
            updated_conversations.append(updated_item)
        else:
            updated_conversations.append(item)
    return {"conversations": updated_conversations}


try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_data = move_metadata(data)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)

    print(f"成功将 metadata 移动并保存到: {output_file_path}")

except FileNotFoundError:
    print(f"错误: 文件未找到: {input_file_path}")
except json.JSONDecodeError:
    print(f"错误: JSON 文件解码失败: {input_file_path}")
except Exception as e:
    print(f"发生错误: {e}")
