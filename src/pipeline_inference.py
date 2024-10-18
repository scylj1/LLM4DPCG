
import torch
import os
from transformers import pipeline
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference")

    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct/") 
    parser.add_argument("--directory", type=str, default="../dataset/test/")
    parser.add_argument("--model_name", type=str, default="test")

    args = parser.parse_args()

    pipe = pipeline("text-generation", model=args.model, torch_dtype=torch.bfloat16, device_map="auto")
    directory = args.directory
    
    for i, filename in enumerate(os.listdir(directory)):
        
        if filename.endswith('.json'):  
            
            filepath = os.path.join(directory, filename)  

            with open(filepath, 'r', encoding='utf-8') as file: 
                data = json.load(file)  
                
                if args.model_name+"-claim" in data:
                    print("exist")
                    continue
                
                claim = data["claims"]
                description = data["full_description"]
                
                instruction = "You are a patent expert. Given the following patent description, generate patent claims. Patent description: " + description # (the output should only include claims in order 1, 2, 3, etc) Ensure the claims include all essential features, the language is clear and unambiguous, the terminologies are used consistently, and features are interconnected and related accurately.
             
                messages = [
                    {"role": "user", "content": instruction},
                ]
                prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=1024, return_full_text=False, do_sample=True, temperature=0.1, top_p=0.95)
                text = outputs[0]["generated_text"]
                
                # print(text)
                
                data[args.model_name+"-claim"] = text.replace("\n", "")
                
            with open(filepath, 'w') as file:
                json.dump(data, file)