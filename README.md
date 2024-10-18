# Large Language Models for Patent Claim Generation

This repository is the code for generating patent claims based on the patent description. It supports fine-tuning and multi-task fine-tuning of the [Llama-3 model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). It also includes the inference code for fine-tuned Llama-3, models on [HuggingFace](https://huggingface.co/models), and GPT-based models from [OpenAI](https://platform.openai.com/docs/models).

## Dataset

Original dataset can be downloaded [here](https://huggingface.co/datasets/lj408/HUPD-DCG).

To prepare dataset for fine-tuning:

```bash
cd src
python data_setup.py --directory [training data directory] --out_path [output directory]
```

## Environment setup

```bash
sh init_env.sh
```

## Training

```bash
cd src
accelerate launch --config_file accelerate_ds_config.yaml pefts/mft_accelerate.py --train_config configs/lora_train_config.json --distributed_type "DeepSpeed" 
```

Change `configs/lora_train_config.json` for specific settings. 

## Inference

Inference of fine-tuned models with LoRA adapters:
```bash
python hf_inference.py --base_model [path to the base model] --peft_path [path to the trained adapter] --directory [test data directory]
```

Inference of models on Huggingface:
```bash
python pipeline_inference.py --model [path to the model] --directory [test data directory]
```

Inference of GPT series from OpenAI:
```bash
python openai_inference.py --openai_key [your key] --model_name [model name] --data_dir [test data directory]
```

## Evaluation

```bash
python eval.py --model_name [model outputs for evaluation] --data_dir [test data directory]
```

## Results

Sample outputs are shown in the `results` folder. 

## Acknowledgement

The code is built based on [MFTCoder](https://github.com/codefuse-ai/MFTCoder/tree/main), a multi-task fine-tuning framework. Detailed instructions of fine-tuning can be found there.

## Citation

If you find our work useful for your research, please feel free to cite our paper.
```
@article{jiang2024can,
  title={Can Large Language Models Generate High-quality Patent Claims?},
  author={Lekang Jiang and Caiqi Zhang and Pascal A Scherz and Stephan Goetz},
  journal={arXiv preprint arXiv:2406.19465},
  year={2024}
}
```