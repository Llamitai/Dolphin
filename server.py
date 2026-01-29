import base64
import io
import os
import time

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from utils.utils import resize_img, parse_layout_string, process_coordinates, check_bbox_overlap

app = FastAPI(title="Dolphin OCR Server")

dolphin = None


class DolphinModel:
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.model.eval()

        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

    def chat(self, prompt, image):
        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        assert len(images) == len(prompts)

        processed_images = [resize_img(img) for img in images]

        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            all_messages.append(messages)

        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]

        all_image_inputs = []
        for msgs in all_messages:
            image_inputs, _ = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        results = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if not is_batch:
            return results[0]
        return results


def process_elements(layout_results, image, model, max_batch_size=4):
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *image.size], "distorted_page", [])]
    elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, image):
        print("Falling back to distorted_page mode due to high bbox overlap")
        layout_results_list = [([0, 0, *image.size], "distorted_page", [])]

    tab_elements = []
    equ_elements = []
    code_elements = []
    text_elements = []
    figure_results = []
    reading_order = 0

    for bbox, label, tags in layout_results_list:
        try:
            if label == "distorted_page":
                x1, y1, x2, y2 = 0, 0, *image.size
                pil_crop = image
            else:
                x1, y1, x2, y2 = process_coordinates(bbox, image)
                pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                if label == "fig":
                    buf = io.BytesIO()
                    pil_crop.save(buf, format="PNG")
                    fig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    figure_results.append({
                        "label": label,
                        "text": "",
                        "figure_base64": fig_b64,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    }
                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()

    if tab_elements:
        recognition_results.extend(
            process_element_batch(tab_elements, model, "Parse the table in the image.", max_batch_size)
        )
    if equ_elements:
        recognition_results.extend(
            process_element_batch(equ_elements, model, "Read formula in the image.", max_batch_size)
        )
    if code_elements:
        recognition_results.extend(
            process_element_batch(code_elements, model, "Read code in the image.", max_batch_size)
        )
    if text_elements:
        recognition_results.extend(
            process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        )

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    results = []
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)

    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i + batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)

        batch_results = model.chat(prompts_list, crops_list)

        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
                "tags": elem["tags"],
            })

    return results


@app.on_event("startup")
async def startup_event():
    global dolphin
    model_path = os.environ.get("MODEL_PATH", "./hf_model")
    print(f"Loading Dolphin model from {model_path}...")
    dolphin = DolphinModel(model_path)
    print(f"Model loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": dolphin is not None}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    layout_output = dolphin.chat("Parse the reading order of this document.", image)

    recognition_results = process_elements(layout_output, image, dolphin)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return JSONResponse({
        "blocks": recognition_results,
        "raw_response": layout_output,
        "width": width,
        "height": height,
        "processing_time_ms": elapsed_ms,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
