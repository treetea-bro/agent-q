import torch
from transformers import TrainerCallback


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self, tokenizer, every_n_steps=500, prompt="Explain quantum computing simply."
    ):
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.prompt = prompt

    def on_step_end(self, args, state, control, **kwargs):
        # N step 마다 실행
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            model = kwargs["model"]
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)

            model.eval()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            model.train()

            print(f"\n[Step {state.global_step}] Prompt: {self.prompt}")
            print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return control


trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[SampleGenerationCallback(tokenizer, every_n_steps=500)],
)
