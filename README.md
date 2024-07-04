# Large Language Models for Patent Claim Generation

This repository is the code for generating patent claims based on the patent description. It supports fine-tuning and multi-task fine-tuning of the [Llama-3 model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). It also includes the inference code for Llama-3 and GPT-based models from [OpenAI](https://platform.openai.com/docs/models).

## Environment setup

```bash
sh init_env.sh
```

## Training

```bash
cd src
accelerate launch --config_file accelerate_ds_config.yaml pefts/mft_accelerate.py --train_config configs/lora_train_config.json --distributed_type "DeepSpeed" 
```

## Inference

For Llama-3 inference:
```bash
python hf_inference.py --base_model [path to the base model] --peft_path [path to the trained adapter] --directory [test data directory]
```

For OpenAI inference:
```bash
python openai_inference.py --openai_key [your key] --model_name [model name] --data_dir [test data directory]
```

## Evaluation

```bash
python eval.py --model_name [model outputs for evaluation] --data_dir [test data directory]
```

## Acknowledgement

The code is built based on [MFTCoder](https://github.com/codefuse-ai/MFTCoder/tree/main), a multi-task fine-tuning framework.

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