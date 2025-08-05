from datasets import load_dataset
        
# 加载 jsonl 文件
dataset = load_dataset("json", data_files="./sunwukong_style_100.jsonl", split="train")
        
# 保存为 GUI 能识别的本地磁盘格式
dataset.save_to_disk("./sunwukong_style_100_for_gui")
print("数据集已转换并保存成功！现在可以在 GUI 中使用了。")
