from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig                                                                                                               
from PIL import Image                                                                      
import torch                                                                                                                                                                                              
                                                                                                                                                                                                        
model_id = "DocTron-Hub/VinciCoder-8B-SFT"
                                                                                                                                                                                                        
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                                                                                                                                                                                                        
processor = AutoProcessor.from_pretrained(model_id)                                                                                                                                                     
model = Qwen2VLForConditionalGeneration.from_pretrained(                                                                                                                                                  
    model_id, quantization_config=bnb_config, device_map="auto"                                                                                                                                     
)                                                              


image = Image.open("/home/raymark/dev/i2c/google.png")
                                                                                                                                                                                                        
messages = [{                                                                                                                                                                                           
    "role": "user",
    "content": [   
        {"type": "image", "image": image},
        {"type": "text", "text": "Convert this image to code."},
    ],                                                          
}]                                                                                                                                                                                                        

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)                                                                                                                
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)                                                                                                                   
                                                                                    
with torch.no_grad():                                                                                                                                                                                     
    output = model.generate(**inputs, max_new_tokens=2048)
                                                                                                                                                                                                        
print(processor.decode(output[0], skip_special_tokens=True))                      