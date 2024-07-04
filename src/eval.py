from evaluate import load as load_metric
import json
import os
import argparse
import tiktoken
import re


def evaluate_text(prediction_texts, reference_texts):
    metrics = {"rouge1": 0, "bleu": 0, "rougeL": 0, "bertscore": 0}

    # BLEU
    references_for_bleu = [[text] for text in reference_texts]
    bleu = load_metric("bleu")
    bleu_result = bleu.compute(predictions=prediction_texts, references=references_for_bleu)
    metrics["bleu"] = bleu_result["bleu"]
    print(metrics["bleu"])
    # ROUGE
    rouge = load_metric("rouge")
    rouge_result = rouge.compute(predictions=prediction_texts, references=reference_texts)
    metrics["rougeL"] = rouge_result["rougeL"]
    metrics["rouge1"] = rouge_result["rouge1"]
    print(metrics["rougeL"])
    print(metrics["rouge1"])
    # BERTScore
    bertscore = load_metric("bertscore")
    results = bertscore.compute(predictions=prediction_texts, references=reference_texts, lang="en", device="cuda:0")
    metrics["bertscore"] = sum(results["f1"]) / len(results["f1"]) 

    return metrics
     
def eval(args):
    reference_text = []
    outputs = []
    directory = args.data_dir
    data_type = args.data_type
    model = args.model_name
    
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.json'):  
            filepath = os.path.join(directory, filename)  
            with open(filepath, 'r', encoding='utf-8') as file: 
                try:
                    data = json.load(file)  
                    
                    if data_type == "claims":
                        text = data[model]
                        outputs.append(text)
                        reference_text.append(text)
                             
                    elif data_type == "abstract":
                        outputs.append(data[model])
                        reference_text.append(data["abstract"])

                    else:
                        print("data type not valid")
                        return 0
                except:
                    print(filepath)
                    continue
    print(f"length of eval list {len(outputs)}")
    scores = evaluate_text(outputs, reference_text)
    print(scores)
    
  

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Openai settings")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="claims")
    parser.add_argument("--data_dir", type=str, default="../../dataset/test")

    args = parser.parse_args()
    
    eval(args)