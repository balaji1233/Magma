# Introduction 
Magma Training Code with native huggingface model support

# Getting Started

## Install dependencies
1. Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
2. Install bug-fixed transformers from the following repository:
```bash
pip install git+https://github.com/jwyang/transformers.git@dev/jwyang
```

## Data Preparation
We are using aitw as the dataset for training sample.

## Init Model Preparation
We are using a pretrainied magma base model as the initialization.

## Run the training sample code
1. Specify the paths in the sample code:
```python
# Specify the paths
# default MODEL_PATH or use the one from the environment
MODEL_PATH="/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma-hf/magma-llama-3-8b-instruct-base-hf"
# default DATA_PATH
DATA_PATH="/home/jianwyan/projects/ProjectWillow/azureblobs/qianhuiwu/seeClick_aitw/v4/texts/all_domains_train_flattened.json"
# default IMAGE_FOLDER
IMAGE_FOLDER="/home/jianwyan/projects/ProjectWillow/azureblobs/qianhuiwu"
# default OUTPUT_DIR
OUTPUT_DIR="./checkpoints/finetune-llava-v1.5-llama-3-8b-820k-aitw_sft_19k_0916"
```

2. Run the sample code using the following command:
```bash
sh ./scrits/finetune_magma-8B-base_aitw_debug.sh
```

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)

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

## Spatial Understanding Evaluation

Evaluating the model on spatial understanding tasks has two steps:

1. First, generate the model predictions for each benchmark by running the following commands. You can replace the following file with the appropriate scripts for corresponding benchmarks. This will be cleaned up and unified.
```bash
python evaluations/spatial_understanding/eval_blink.py --model_path {path to model weights} --output_dir {path to output directory}
```

2. Next, to obtain the accuracy based on the generated predictions, run the following command:
```bash
python evaluations/spatial_understanding/eval_mc_qa.py --dataset_name {e.g., blink} ----eval_predictions {path to output directory}
```
