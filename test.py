from PIL import Image
import requests
from transformers import AutoTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device)
pretrained_ckpt = "MAGAer13/mplug-owl-llama-7b"
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)
tokenizer = processor.tokenizer
model = MplugOwlForConditionalGeneration.from_pretrained(pretrained_ckpt, torch_dtype=torch.bfloat16)
model.to(device)  # doctest: +IGNORE_RESULT

# LoRA 
# peft_config = LoraConfig(target_modules=r'.*language_model.*\.(q_proj|v_proj)', inference_mode=False, r=8,lora_alpha=32, lora_dropout=0.05)
# self.model = get_peft_model(self.model, peft_config)
# lora_path = 'Your lora model path'
# prefix_state_dict = torch.load(lora_path, map_location='cpu')
# self.model.load_state_dict(prefix_state_dict)

while True:
    url = input('Image url:')
    image = Image.open(requests.get(str(url), stream=True).raw)
    request = input('Request:')
    prompt = [
      "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: "+ str(request) + "\nAI: "
      ]
    inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(sequences=generated_ids)[0].strip()
    print("Description: " + generated_text)



        
