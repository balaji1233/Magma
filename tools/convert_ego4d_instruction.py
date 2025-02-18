import torch
import pickle


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    flash_attn=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def convert(model, tokenizer, narration):

    messages = [
        {"role": "system", "content": "You are a language expert. Given a sentence, you can convert it to a instruction format. For example, 'now I want to cut the apple' should be converted to 'cut the apple'. Do not include any transition words."},
        {"role": "user", "content": f"{narration}"},
    ]    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        attention_mask=input_ids.ne(0),
        max_new_tokens=512,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

ego4d_ann = "/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/video_datasets_visual_traces/ego4d/language_annotations/cleaned_vid_to_anns.pkl"

lang_dict = pickle.load(open(ego4d_ann, "rb"))              

for key, val in tqdm(lang_dict.items()):
    # print(key, val)
    for k, v in val.items():
        v_instruction = convert(model, tokenizer, v)
        val[k] = v_instruction
import pdb; pdb.set_trace()
