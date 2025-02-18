# Magma: Multimodal Agentic Models

Magma: A multimodal agentic foundation for multimodal understanding and agentic tasks.

## Introduction

Magma is a multimodal agentic AI model that can generate text based on the input text and image. The model is designed for research purposes and aimed at knowledge-sharing and accelerating research in multimodal AI, in particular the multimodal agentic AI. The main innovation of this model lies on the introduction of two technical innovations: Set-of-Mark and Trace-of-Mark, and the leverage of a large-amount of unlabeled video data to learn the spatial-temporal grounding and planning. Please refer to our paper for more technical details. The model is developed by Microsoft and is funded by Microsoft Research. 

## Model Usage

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is intended for broad research use in English. The model take images and text as inputs, and produces the textual outputs for the following uses:

* **Image/Video-Conditoned Text Generation:** The model can generate text (e.g., descriptions, answers) based on the input text and image.

* **Visual Planning Capabilities:** The model can also produce the visual trace as the future planning to accomplish a task (e.g., move object from one place to another).

* **Agentic Capabilities:** The model can also generate UI grounding (e.g., click ``search'' button) and robotics manipulations (e.g., 7 DoF for the robot gripper).

Our model is designed only for research purpose and aimed at knowledge-sharing and accelerating research in multimodal AI, in particularly the mutimodal agentic AI.

### Downstream Use

The model can be further finetuned for different downstream tasks, such as:

* **Image Captioning and QA:** We can further finetune this model for image captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better spatial understanding and reasoning on these tasks.

* **Video Captioning and QA:** We can further finetune this model for video captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better temporal understanding and reasoning on these tasks.

* **UI Navigation:** We can finetune this model for specific UI navigation tasks, such as web navigation or mobile navigation. The model can achieve superior performance on these tasks.

* **Robotics Manipulation:** Our model can be further finetuned for robotics tasks given its general agentic capabilities as a vision-language-action model. After finetuning, our model significantly outperms the state-of-the-art models such as OpenVLA on robotics manipulation tasks.

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

<!-- {{ downstream_use | default("[More Information Needed]", true)}} -->

<!-- ### Out-of-Scope Use -->

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

<!-- {{ out_of_scope_use | default("[More Information Needed]", true)}} -->

### Quick Start for Testing Magma in Manipulation

This quick guide helps you evaluate Magma's performance on manipulation tasks in LIBERO. 

#### LIBERO Setup
Clone and install LIBERO and other requirements:
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -r libero_quick_start/requirements.txt
cd LIBERO
pip install -e .
```

#### Quick Evaluation
The following code demonstrates how to run Magma on a single LIBERO task and evaluate its performance:
```
import numpy as np
from libero.libero import benchmark
from libero_eval.libero_env_utils import get_libero_env, get_libero_dummy_action, get_libero_obs, get_max_steps, save_rollout_video
from libero_eval.libero_magma_utils import get_magma_model, get_magma_prompt, get_magma_action

# Set up benchmark and task
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_goal" # or libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
task_id = 1
task = task_suite.get_task(task_id)

# Initialize environment
env, task_description = get_libero_env(task, resolution=256)
print(f"Task {task_id} description: {task_description}")

# Load MAGMA model
model_name = "microsoft/magma-8b-libero-goal"  # or your local path
processor, magma = get_magma_model(model_name)
prompt = get_magma_prompt(task_description)

# Run evaluation
num_steps_wait = 10
max_steps = get_max_steps(task_suite_name)

env.seed(0)
obs = env.reset()
init_states = task_suite.get_task_init_states(task_id) 
obs = env.set_init_state(init_states[0])

step = 0
replay_images = []
while step < max_steps + num_steps_wait:
    if step < num_steps_wait:
        obs, _, done, _ = env.step(get_libero_dummy_action())
        step += 1
        continue
    
    img = get_libero_obs(obs, resize_size=256)
    replay_images.append(img)
    action = get_magma_action(magma, processor, img, prompt, task_suite_name)
    obs, _, done, _ = env.step(action.tolist())
    step += 1

env.close()
save_rollout_video(replay_images, success=done, task_description=task_description)
```
**Notes:** The above script only tests one episode on a single task and visualizes MAGMA's trajectory with saved video. For comprehensive evaluation on each task suite, please use `libero_eval/eval_magma_libero.py`.
```
python libero_eval/eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-object \
  --task_suite_name libero_object \

python libero_eval/eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-spatial \
  --task_suite_name libero_spatial \

python libero_eval/eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-goal \
  --task_suite_name libero_goal \
```

## Bias, Risks, and Limitations

Please note that this model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
