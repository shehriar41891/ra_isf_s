import time

from openai import OpenAI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_init(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use float16 if on GPU, otherwise float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def predict(args, prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    generate_ids = model.generate(
        **inputs, 
        max_length=args.max_length,  
        temperature=args.temperature  
    )
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):]  
    infer_res = tokenizer.decode(generate_ids, skip_special_tokens=True)  
    
    print('the result of infer from predict function is ',infer_res)
    
    return infer_res


# def predict(args, prompt):
#     my_key = args.api_key
#     max_length = 256
#     temperature = 0.0
#     top_p = 1
#     frequency_penalty = 0
#     presence_penalty = 0
#     client = OpenAI(api_key = my_key)
#     prompt = "
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct", # text-davinci-003 is deprecated
#         prompt=prompt,
#         max_tokens=max_length,
#         temperature=temperature,
#         top_p=top_p,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         #   api_key=my_key,
#     )
#     if args.engine == 'llama2-13b':
#         raise NotImplementedError('Engine false when running gpt3.5: {}'.format(args.engine))
#     return response.choices[0].text