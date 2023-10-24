from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def generate_prompt(instruction, input=None):
    if input:
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"


tokenizer = GPT2Tokenizer.from_pretrained('Midkey/GPT2-3.5B-chinese-ft-luotuo')

model = GPT2LMHeadModel.from_pretrained('Midkey/GPT2-3.5B-chinese-ft-luotuo', trust_remote_code=True, load_in_8bit=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def evaluate(
        instruction,
        input=None,
        max_new_tokens=256,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    eos_token_id = 50256
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=eos_token_id
        )
    for s in generation_output.sequences:
        decode_s = s[len(input_ids[0]):]
        if decode_s[-1] == eos_token_id:
            decode_s = decode_s[:-1]
        output = tokenizer.decode(decode_s)
        print("Response:", output)


instruction = '华中师范大学在什么地方?'
evaluate(instruction)