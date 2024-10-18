import os
import json
import argparse


def create_train(args):
    
    directory = args.directory
    results = []

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.json'):  
            filepath = os.path.join(directory, filename)  
            with open(filepath, 'r', encoding='utf-8') as file: 
                data = json.load(file)  

                result = {  "id": i,
                            "data_name":"patent-helper",
                            "chat_rounds":[
                                {
                                    "role": "system",
                                    "content": "You are a patent expert and you can generate patent texts precisely."
                                },
                                {
                                    "role": "human",
                                    "content": "Given the following patent description, generate patent claims. Patent description: " + data["full_description"]
                                },
                                {
                                    "role": "bot",
                                    "content": data["claims"]
                                },
                   
                            ]
                         }              
                
                results.append(result)

    with open(args.out_path + 'train.jsonl', 'w') as file:
        for entry in results:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

    print(f"Data has been written to {args.out_path}")
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Date setup")

    parser.add_argument("--directory", type=str, default="../dataset/train")
    parser.add_argument("--out_path", type=str, default="../dataset/train_description_claim")
  
    args = parser.parse_args()

    create_train(args)