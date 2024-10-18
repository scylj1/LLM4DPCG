import json
from openai import OpenAI
import os
import tiktoken
import argparse


def openai_inference(args):
    model = args.model_name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", args.openai_key))
    directory = args.data_dir

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.json'):  
            filepath = os.path.join(directory, filename)  
            with open(filepath, 'r', encoding='utf-8') as file: 
                data = json.load(file)  
                
                if model+"-claim" in data:
                    print(filepath + " exist")
                    continue 
                
                description = data["full_description"]  
    
                messages=[{"role": "system", "content": "You are a patent expert. Given the following patent description, generate patent claims. "}]    
                messages.append({"role": "user", "content": description})
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,
                  
                    )

                    text = response.choices[0].message.content
                    text = text.replace("\n", " ")

                    data[model+"-claim"] = text
                                    
                except Exception as e:
                    
                    print("An error occurred:", e)
                    print(data["application_number"])
                    continue

            with open(filepath, 'w') as file:
                json.dump(data, file)
            
            if i % 50 == 0:
                print(f"{i} documents complete")    

        
if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Openai settings")

    parser.add_argument("--model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="../dataset/test/")

    args = parser.parse_args()

    openai_inference(args)
  