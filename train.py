from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# 모델 및 토크나이저
model_name = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 학습/검증 데이터셋 (chosen, rejected 쌍이 있는 데이터셋 필요)
dataset = load_dataset("Anthropic/hh-rlhf")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # 예시로 1000개
eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))  # 평가용 200개

# 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

# Reference model (학습 전 모델 그대로)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

# 학습 config
training_args = DPOConfig(
    output_dir="./qwen3-30b-dpo",
    per_device_train_batch_size=1,  # 30B는 GPU 메모리 많이 차지하므로 batch 작게
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=1,
    fp16=True,
)

# Trainer 생성
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 학습 후 모델 저장
trainer.save_model("./qwen3-30b-dpo")
tokenizer.save_pretrained("./qwen3-30b-dpo")
