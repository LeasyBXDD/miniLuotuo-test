from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 指定预训练模型的名称
model_name_or_path = 'gpt2-3.5B'

# 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# 加载训练数据
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="MyTrain.txt",
    block_size=128
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-3.5B-HZNUwlaq", # 输出目录
    overwrite_output_dir=True, # 覆盖输出目录
    num_train_epochs=3, # 训练轮数
    per_device_train_batch_size=32, # 每个设备的训练批次大小
    save_steps=10_000, # 保存模型的步数
    save_total_limit=2, # 保存模型的总限制
)

# 设置训练器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()