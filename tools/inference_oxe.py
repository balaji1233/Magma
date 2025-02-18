import torch
import torchvision
from PIL import Image
from magma.utils.conversation import conv_templates
from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForConditionalGeneration

model_id = "/home/jianwyan/projects/ProjectWillow/azureblobs/projects4jw_model/magma/checkpoints/finetune-none-bs8-ep5-bimsz512-ncrops4-anyresglobal-seqlen2048-1e-5-constant-0.0_epic_coin_ego4d_openx_-1_iseFalse_tseFalse_tsdTrue_rtptsTrue_qsz256/checkpoint-5000"
processor = MagmaProcessor.from_pretrained(model_id, trust_remote_code=True) 
model = MagmaForConditionalGeneration.from_pretrained(
	model_id, 
	device_map="cuda", 
	low_cpu_mem_usage=True
) # use _attn_implementation='eager' to disable flash attention

conv_mode = "llama_3_instruct"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], "<image>\nWhat action should the robot take to move blue plastic bottle near pepsi can?")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open("temp_oxe.jpg")
inputs = processor(images=image, texts=prompt, return_tensors="pt")
inputs = inputs.to("cuda")

# verify generation
output_ids = model.generate(
  **inputs,
  temperature=0.2,
  do_sample=True,
  num_beams=1,
  max_new_tokens=1000,
  use_cache=True,
)

import pdb; pdb.set_trace()
# only decode generated text
prompt_decoded = processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0]
generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# remove prompt from generated text
response = generated_text[len(prompt_decoded):]
print("Generated text:", repr(response))