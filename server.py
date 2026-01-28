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
                                                                                                        
from utils.utils import resize_img                                                                     
                                                                                                        
app = FastAPI(title="Dolphin OCR Server")                                                              
                                                                                                        
model = None                                                                                           
processor = None                                                                                       
                                                                                                        
                                                                                                        
def load_model():                                                                                      
    global model, processor                                                                            
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
            max_new_tokens=8192,                                                                       
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
                                                                                                        
    prompt = """Parse this document and extract all text with their bounding boxes.                    
For each text element, output in this exact format:                                                    
[x1,y1,x2,y2][type]text content here[/type][PAIR_SEP]                                                  
                                                                                                        
Where type is one of: para, title, fig, table, list, foot, head, caption, equation                     
                                                                                                        
Extract ALL text visible in the document."""                                                           
                                                                                                        
    response = chat(prompt, image)                                                                     
    elapsed_ms = (time.perf_counter() - start_time) * 1000                                             
                                                                                                        
    blocks = parse_response(response, width, height)                                                   
                                                                                                        
    return JSONResponse({                                                                              
        "blocks": blocks,                                                                              
        "raw_response": response,                                                                      
        "width": width,                                                                                
        "height": height,                                                                              
        "processing_time_ms": elapsed_ms,                                                              
    })                                                                                                 
                                                                                                        
                                                                                                        
def parse_response(response: str, width: int, height: int) -> list:                                    
    """Parse Dolphin response format: [x1,y1,x2,y2][type]text[/type][PAIR_SEP]..."""                   
    blocks = []                                                                                        
                                                                                                        
    type_map = {                                                                                       
        "para": "paragraph",                                                                           
        "title": "title",                                                                              
        "fig": "figure",                                                                               
        "table": "table",                                                                              
        "list": "list",                                                                                
        "foot": "footer",                                                                              
        "head": "header",                                                                              
        "caption": "caption",                                                                          
        "equation": "formula",                                                                         
    }                                                                                                  
                                                                                                        
    parts = response.split("[PAIR_SEP]")                                                               
                                                                                                        
    for i, part in enumerate(parts):                                                                   
        part = part.strip()                                                                            
        if not part:                                                                                   
            continue                                                                                   
                                                                                                        
        # Pattern with text: [x1,y1,x2,y2][type]text content[/type]                                    
        pattern_with_text = r'\[(\d+),(\d+),(\d+),(\d+)\]\[(\w+)\](.*?)\[/\5\]'                        
        match = re.match(pattern_with_text, part, re.DOTALL)                                           
                                                                                                        
        if match:                                                                                      
            x1, y1, x2, y2, block_type, text = match.groups()                                          
            text = text.strip()                                                                        
        else:                                                                                          
            # Fallback: try pattern without closing tag                                                
            pattern_simple = r'\[(\d+),(\d+),(\d+),(\d+)\]\[(\w+)\](.*?)$'                             
            match = re.match(pattern_simple, part, re.DOTALL)                                          
                                                                                                        
            if match:                                                                                  
                x1, y1, x2, y2, block_type, text = match.groups()                                      
                text = text.strip()                                                                    
            else:                                                                                      
                # Original pattern without text                                                        
                pattern_no_text = r'\[(\d+),(\d+),(\d+),(\d+)\]\[(\w+)\]'                              
                match = re.match(pattern_no_text, part)                                                
                if match:                                                                              
                    x1, y1, x2, y2, block_type = match.groups()                                        
                    text = ""                                                                          
                else:                                                                                  
                    continue                                                                           
                                                                                                        
        normalized_type = type_map.get(block_type.lower(), block_type.lower())                         
                                                                                                        
        blocks.append({                                                                                
            "id": i,                                                                                   
            "type": normalized_type,                                                                   
            "text": text,                                                                              
            "bbox": [int(x1), int(y1), int(x2), int(y2)],                                              
            "confidence": 1.0,                                                                         
        })                                                                                             
                                                                                                        
    return blocks                                                                                      
                                                                                                        
                                                                                                        
if __name__ == "__main__":                                                                             
    import uvicorn                                                                                     
    uvicorn.run(app, host="0.0.0.0", port=8000)   
