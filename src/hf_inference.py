
import os
import sys
import torch
import textwrap
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import json
import argparse

def load_model_tokenizer(path, model_type=None, peft_path=None, torch_dtype=torch.bfloat16, quantization=None,
                         eos_token=None, pad_token=None):
    """
        load model and tokenizer by transfromers
    """

    # load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    #tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    config, unused_kwargs = AutoConfig.from_pretrained(
        path,
        trust_remote_code=True,
        return_unused_kwargs=True
    )
    print("unused_kwargs:", unused_kwargs)
    print("config input:\n", config)

    # eos token优先级: 1. 用户输入eos_token 2. config中的eos_token_id 3. config中的eos_token
    if eos_token:
        eos_token = eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        print(f"eos_token {eos_token} from user input")
    else:
        if hasattr(config, "eos_token_id") and config.eos_token_id:
            print(f"eos_token_id {config.eos_token_id} from config.json")
            eos_token_id = config.eos_token_id
            eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        elif hasattr(config, "eos_token") and config.eos_token:
            print(f"eos_token {config.eos_token} from config.json")
            eos_token = config.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
        else:
            raise ValueError(
                "No available eos_token or eos_token_id, please provide eos_token by params or eos_token_id by config.json")

    # pad token优先级: 1. 用户输入 pad_token 2. config中的pad_token_id 3. config中的pad_token
    if pad_token:
        pad_token = pad_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        print(f"pad_token {pad_token} from user input")
    else:
        if hasattr(config, "pad_token_id") and config.pad_token_id:
            print(f"pad_token_id {config.pad_token_id} from config.json")
            pad_token_id = config.pad_token_id
            pad_token = tokenizer.convert_ids_to_tokens(config.pad_token_id)
        elif hasattr(config, "pad_token") and config.pad_token:
            print(f"pad_token {config.pad_token} from config.json")
            pad_token = config.pad_token
            pad_token_id = tokenizer.convert_tokens_to_ids(config.pad_token)
        else:
            print(f"pad_token {eos_token} duplicated from eos_token")
            pad_token = eos_token
            pad_token_id = eos_token_id

    # update tokenizer eos_token and pad_token
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token
    tokenizer.pad_token_id = pad_token_id
    tokenizer.pad_token = pad_token

    print(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    print(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    print(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        load_in_8bit=(quantization == '8bit'),
        load_in_4bit=(quantization == '4bit'),
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if peft_path:
        print("Loading PEFT MODEL...")
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        print("Loading Original MODEL...")
        model = base_model

    model.eval()

    print("=======================================MODEL Configs=====================================")
    print(model.config)
    print("=========================================================================================")
    print("=======================================MODEL Archetecture================================")
    print(model)
    print("=========================================================================================")

    return model, tokenizer


def hf_inference(model, tokenizer, text_list, args=None, max_new_tokens=512, do_sample=True, **kwargs):
    """
        transformers models inference by huggingface
    """
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
    # inputs["attention_mask"][0][:100] = 0
    # print(inputs)
    print("================================Prompts and Generations=============================")

    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )

    gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for i in range(len(text_list)):
        print('=========' * 10)
        print(f'Prompt:\n{text_list[i]}')
        gen_text[i] = gen_text[i].replace(tokenizer.pad_token, '')
        print(f'Generation:\n{gen_text[i]}')
        # print(f"Outputs ids:\n{outputs[i]}")
        sys.stdout.flush()

    return gen_text


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="inference")

    parser.add_argument("--base_model", type=str, default="../../../Meta-Llama-3-8B-Instruct-hf/")
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--directory", type=str, default="../dataset/test")
    parser.add_argument("--model_name", type=str, default="test")
  

    args = parser.parse_args()
    
    # Default template used in MFTCoder training
    HUMAN_ROLE_START_TAG = "<s>human\n"
    BOT_ROLE_START_TAG = "<s>bot\n"

    # if you use base + adaptor for inference, provide peft_path or left it None for normal inference
    base_model = args.base_model
    peft_path = args.peft_path
    model, tokenizer = load_model_tokenizer(base_model,
                                            model_type='',
                                            peft_path=peft_path,
                                            eos_token='<|eot_id|>',
                                            pad_token='<pad>')

    directory = args.directory

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.json'):  
            filepath = os.path.join(directory, filename)  
            with open(filepath, 'r', encoding='utf-8') as file: 
                data = json.load(file)  

                description = data["full_description"]
                
                instruction = "Given the following patent description, generate patent claims. Patent description: " + description
                prompts = [f"{HUMAN_ROLE_START_TAG}{instruction}\n{BOT_ROLE_START_TAG}"]
                text = hf_inference(model, tokenizer, prompts, do_sample=True, temperature=0.1, max_new_tokens=1024)
                
                data[args.model_name+"-claim"] = text
        
            with open(filepath, 'w') as file:
                json.dump(data, file)
                
