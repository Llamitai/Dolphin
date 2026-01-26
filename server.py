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

    prompt = """Please parse the reading order of this document. Follow the order of human reading, from top to bottom, and from left to right.
You need to output the content of all text blocks, following the format:
<BLOCK id=x type=TYPE>
the text content of this block
</BLOCK>
Possible values for TYPE are: paragraph, title, subtitle, header, footer, table, figure, caption, equation, list.
"""

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
    blocks = []
    pattern = r'<BLOCK\s+id=(\d+)\s+type=(\w+)>(.*?)</BLOCK>'
    matches = re.findall(pattern, response, re.DOTALL)

    for block_id, block_type, content in matches:
        blocks.append({
            "id": int(block_id),
            "type": block_type.lower(),
            "text": content.strip(),
            "bbox": [0, 0, width, height],
            "confidence": 1.0,
        })

    return blocks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
