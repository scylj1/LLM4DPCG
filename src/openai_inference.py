import json
from openai import OpenAI
import os
import tiktoken
import argparse


def openai_inference(args):
    model = args.model_name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", args.openai_key))
    directory = args.data_dir
    fail_list = []

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.json'):  
            filepath = os.path.join(directory, filename)  
            with open(filepath, 'r', encoding='utf-8') as file: 
                data = json.load(file)  
                
                if model+"-claims" in data:
                    print(filepath + " exist")
                    continue 
                
                description = data["full_description"]  
    
                messages=[{"role": "system", "content": "You are a patent expert. Given the following patent description, write patent claims and patent abstract. The answer should be in the format of: Patent claims: generated patent claims. ##seperator## Patent abstract: generated patent abstract. "}]    
                messages.append({"role": "user", "content": description})
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,
                  
                    )

                    text = response.choices[0].message.content
                    text = text.replace("\n", " ")
                    separator = "##seperator##"
                    parts = text.split(separator)

                    claims = parts[0]
                    abstract = parts[1]
                    data[model+"-claims"] = claims
                    data[model+"-astract"] = abstract
           
                    print(data["application_number"])
                    print(data[model+"-claims"])
                    print(response.usage.completion_tokens)
                    # print(data[model+"-astract"])
                                    
                except Exception as e:
                    
                    print("An error occurred:", e)
                    print("fail")
                    print(data["application_number"])
                    fail_list.append(data["application_number"])
                    continue

            with open(filepath, 'w') as file:
                json.dump(data, file)
            
            if i % 50 == 0:
                print(f"{i} documents complete")    
        
    with open("fail_list.json", 'w') as file:
        json.dump(fail_list, file)

        
if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Openai settings")

    parser.add_argument("--model_name", type=str, default="gpt-4-turbo")
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="../dataset/test/")

    args = parser.parse_args()

    openai_inference(args)
  