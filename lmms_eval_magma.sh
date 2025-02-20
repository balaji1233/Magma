# login to huggingface if needed

eval_tasks=${1:-videomme}

python3 -m accelerate.commands.launch --main_process_port=29500 --num_processes=4 -m lmms_eval --model magma --model_args pretrained="/path/to/trained/weights" \
    --tasks $eval_tasks --batch_size 1 --log_samples --log_samples_suffix log_file --output_path ./logs/