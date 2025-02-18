# Introduction 
Magma Training Code with native huggingface model support

# Getting Started

## Data Preparation
Download the LLaVa-Video-178K dataset into a folder and specify the path in data_configs.video_instruction_tuning_cluster.yaml.

## Init Model Preparation
You can download the pretrained Magma model weights to initialize as the base weights for video supervised fine-tuning. 

## Run the training sample code
2. Run the sample code using the following command:
```bash
sh ./scripts/magma_video_sft.sh
```

## Evaluation

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate the model. 

To set up the evaluation code for Magma, first clone the repo into the root folder:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

After that, you need to copy two files to lmms-eval/lmms-eval/models:
```bash
cp ./lmms-eval-magma/magma.py lmms-eval/lmms-eval/models/
cp ./lmms-eval-magma/__init__.py lmms-eval/lmms-eval/models/
```

Then you can run the evaluation code:
```bash
sh scripts/lmms_eval_magma_1node.sh
```
