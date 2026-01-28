import io                                                                                              
import os                                                                                              
import re                                                                                              
import time                                                                                            
                                                                                                        
import torch                                                                                           
from PIL import Image                                                                                  
from fastapi import FastAPI, File, UploadFile                                                          
from fastapi.responses import JSONResponse                                                             
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig         
from qwen_vl_utils import process_vision_info                                                          
                                                                                                        
from utils.utils import resize_img, parse_layout_string, process_coordinates                           
                                                                                                        
app = FastAPI(title="Dolphin OCR Server")                                                              
                                                                                                        
model = None                                                                                           
processor = None                                                                                       
tokenizer = None                                                                                       
                                                                                                        
                                                                                                        
def load_model():                                                                                      
    global model, processor, tokenizer                                                                 
    model_path = os.environ.get("MODEL_PATH", "./hf_model")                                            
    print(f"Loading Dolphin model from {model_path}...")                                               
                                                                                                        
    processor = AutoProcessor.from_pretrained(model_path)                                              
                                                                                                        
    quantization_config = BitsAndBytesConfig(                                                          
        load_in_4bit=True,                                                                             
        bnb_4bit_compute_dtype=torch.bfloat16,                                                         
        bnb_4bit_quant_type="nf4",                                                                     
    )                                                                                                  
                                                                                                        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(                                        
        model_path,                                                                                    
        quantization_config=quantization_config,                                                       
        device_map="auto",                                                                             
    )                                                                                                  
    model.eval()                                                                                       
                                                                                                        
    tokenizer = processor.tokenizer                                                                    
    tokenizer.padding_side = "left"                                                                    
                                                                                                        
    print(f"Model loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")                         
                                                                                                        
                                                                                                        
def chat(prompt: str, image: Image.Image) -> str:                                                      
    processed_image = resize_img(image)                                                                
    messages = [                                                                                       
        {                                                                                              
            "role": "user",                                                                            
            "content": [                                                                               
                {"type": "image", "image": processed_image},                                           
                {"type": "text", "text": prompt}                                                       
            ],                                                                                         
        }                                                                                              
    ]                                                                                                  
                                                                                                        
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)         
    image_inputs, _ = process_vision_info(messages)                                                    
                                                                                                        
    inputs = processor(                                                                                
        text=[text],                                                                                   
        images=image_inputs if image_inputs else None,                                                 
        padding=True,                                                                                  
        return_tensors="pt",                                                                           
    )                                                                                                  
    inputs = inputs.to(model.device)                                                                   
                                                                                                        
    with torch.no_grad():                                                                              
        generated_ids = model.generate(                                                                
            **inputs,                                                                                  
            max_new_tokens=4096,                                                                       
            do_sample=False,                                                                           
        )                                                                                              
                                                                                                        
    generated_ids_trimmed = [                                                                          
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)              
    ]                                                                                                  
    output = processor.batch_decode(                                                                   
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False            
    )                                                                                                  
    return output[0]                                                                                   
                                                                                                        
                                                                                                        
@app.on_event("startup")                                                                               
async def startup_event():                                                                             
    load_model()                                                                                       
                                                                                                        
                                                                                                        
@app.get("/health")                                                                                    
async def health():                                                                                    
    return {"status": "ok", "model_loaded": model is not None}                                         
                                                                                                        
                                                                                                        
@app.post("/extract")                                                                                  
async def extract(file: UploadFile = File(...)):                                                       
    start_time = time.perf_counter()                                                                   
                                                                                                        
    image_bytes = await file.read()                                                                    
    image = Image.open(io.BytesIO(image_bytes))                                                        
    if image.mode != "RGB":                                                                            
        image = image.convert("RGB")                                                                   
                                                                                                        
    width, height = image.size                                                                         
                                                                                                        
    # Stage 1: Layout parsing                                                                          
    layout_output = chat("Parse the reading order of this document.", image)                           
    layout_results_list = parse_layout_string(layout_output)                                           
                                                                                                        
    if not layout_results_list or not (layout_output.startswith("[") and layout_output.endswith("]")): 
        layout_results_list = [([0, 0, width, height], "distorted_page", [])]                          
                                                                                                        
    # Stage 2: Extract text from each block                                                            
    blocks = []                                                                                        
    all_texts = []                                                                                     
                                                                                                        
    for i, (bbox, label, tags) in enumerate(layout_results_list):                                      
        try:                                                                                           
            if label == "distorted_page":                                                              
                x1, y1, x2, y2 = 0, 0, width, height                                                   
                pil_crop = image                                                                       
            else:                                                                                      
                x1, y1, x2, y2 = process_coordinates(bbox, image)                                      
                pil_crop = image.crop((x1, y1, x2, y2))                                                
                                                                                                        
            if pil_crop.size[0] <= 3 or pil_crop.size[1] <= 3:                                         
                continue                                                                               
                                                                                                        
            if label == "fig":                                                                         
                text = "[Figure]"                                                                      
            elif label == "tab":                                                                       
                text = chat("Parse the table in the image.", pil_crop)                                 
            elif label == "equ":                                                                       
                text = chat("Read formula in the image.", pil_crop)                                    
            elif label == "code":                                                                      
                text = chat("Read code in the image.", pil_crop)                                       
            else:                                                                                      
                text = chat("Read text in the image.", pil_crop)                                       
                                                                                                        
            text = text.strip() if text else ""                                                        
            all_texts.append(text)                                                                     
                                                                                                        
            blocks.append({                                                                            
                "id": i,                                                                               
                "type": label,                                                                         
                "text": text,                                                                          
                "bbox": [int(x1), int(y1), int(x2), int(y2)],                                          
                "confidence": 1.0,                                                                     
            })                                                                                         
                                                                                                        
        except Exception as e:                                                                         
            print(f"Error processing block {label}: {e}")                                              
            continue                                                                                   
                                                                                                        
    elapsed_ms = (time.perf_counter() - start_time) * 1000                                             
                                                                                                        
    return JSONResponse({                                                                              
        "blocks": blocks,                                                                              
        "raw_response": layout_output,                                                                 
        "width": width,                                                                                
        "height": height,                                                                              
        "processing_time_ms": elapsed_ms,                                                              
    })                                                                                                 
                                                                                                        
                                                                                                        
if __name__ == "__main__":                                                                             
    import uvicorn                                                                                     
    uvicorn.run(app, host="0.0.0.0", port=8000)  
    