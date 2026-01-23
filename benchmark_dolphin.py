#!/usr/bin/env python
"""
Dolphin benchmark script - outputs results in compatible format with documente-core benchmarks.

Usage:
    python benchmark_dolphin.py --model_path ./hf_model --input_path /app/demo/test.png --output_dir /app/demo/results
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from utils.utils import resize_img, parse_layout_string, process_coordinates


class DolphinBenchmark:
    def __init__(self, model_path):
        print(f"Loading Dolphin model from {model_path}...")
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        print(f"Model loaded on {self.device}")

    def chat(self, prompt, image):
        is_batch = isinstance(image, list)
        images = [image] if not is_batch else image
        prompts = [prompt] if not is_batch else (prompt if isinstance(prompt, list) else [prompt] * len(images))

        processed_images = [resize_img(img) for img in images]
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question}
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

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        results = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return results[0] if not is_batch else results

    def extract_text_from_image(self, image: Image.Image) -> tuple[str, list[dict], float]:
        """Extract text from single image, return (full_text, blocks, processing_time_ms)"""
        start_time = time.perf_counter()

        # Stage 1: Layout parsing
        layout_output = self.chat("Parse the reading order of this document.", image)
        layout_results_list = parse_layout_string(layout_output)

        if not layout_results_list or not (layout_output.startswith("[") and layout_output.endswith("]")):
            layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]

        # Stage 2: Extract text from each element
        all_texts = []
        blocks = []

        for reading_order, (bbox, label, tags) in enumerate(layout_results_list):
            try:
                if label == "distorted_page":
                    x1, y1, x2, y2 = 0, 0, *image.size
                    pil_crop = image
                else:
                    x1, y1, x2, y2 = process_coordinates(bbox, image)
                    pil_crop = image.crop((x1, y1, x2, y2))

                if pil_crop.size[0] <= 3 or pil_crop.size[1] <= 3:
                    continue

                if label == "fig":
                    text = "[Figure]"
                elif label == "tab":
                    text = self.chat("Parse the table in the image.", pil_crop)
                elif label == "equ":
                    text = self.chat("Read formula in the image.", pil_crop)
                elif label == "code":
                    text = self.chat("Read code in the image.", pil_crop)
                else:
                    text = self.chat("Read text in the image.", pil_crop)

                text = text.strip() if text else ""
                all_texts.append(text)

                blocks.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "block_type": label,
                    "text": text,
                    "confidence": 1.0,  # Dolphin doesn't provide confidence scores
                })

            except Exception as e:
                print(f"Error processing element {label}: {e}")
                continue

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        full_text = "\n".join(all_texts)

        return full_text, blocks, elapsed_ms


def load_tiff_pages(tiff_path: str) -> list[Image.Image]:
    """Load all pages from a multi-page TIFF."""
    img = Image.open(tiff_path)
    pages = []
    n_frames = getattr(img, "n_frames", 1)
    for i in range(n_frames):
        img.seek(i)
        page = img.copy()
        if page.mode != "RGB":
            page = page.convert("RGB")
        pages.append(page)
    return pages


def run_benchmark(model_path: str, input_path: str, output_dir: str, max_pages: int = None):
    """Run Dolphin benchmark on input file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DolphinBenchmark(model_path)

    input_path = Path(input_path)
    doc_name = input_path.stem

    # Load images
    ext = input_path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        images = load_tiff_pages(str(input_path))
    elif ext in [".png", ".jpg", ".jpeg"]:
        images = [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if max_pages:
        images = images[:max_pages]

    print(f"Processing {len(images)} pages from {input_path.name}")

    # Process each page
    pages_results = []
    total_start = time.perf_counter()

    for i, img in enumerate(images):
        print(f"  Page {i + 1}/{len(images)}...")
        width, height = img.size
        text, blocks, proc_time = model.extract_text_from_image(img)

        pages_results.append({
            "page_index": i,
            "text": text,
            "confidence": 1.0,
            "blocks": blocks,
            "width": width,
            "height": height,
            "processing_time_ms": proc_time,
        })
        print(f"    -> {len(blocks)} blocks, {proc_time:.0f}ms")

    total_time = (time.perf_counter() - total_start) * 1000

    # Build result in compatible format
    result = {
        "document_path": str(input_path),
        "engine_name": "Dolphin",
        "engine_version": "v2_4bit",
        "total_pages": len(pages_results),
        "pages": pages_results,
        "total_processing_time_ms": total_time,
        "metadata": {
            "model_path": model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    }

    # Save result
    result_path = output_dir / f"{doc_name}_dolphin.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {result_path}")
    print(f"Total: {len(pages_results)} pages, {total_time:.0f}ms")

    # Print summary
    avg_conf = sum(p["confidence"] for p in pages_results) / len(pages_results) if pages_results else 0
    print(f"\n{'=' * 60}")
    print(f"DOLPHIN BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Document: {doc_name}")
    print(f"Pages: {len(pages_results)}")
    print(f"Total time: {total_time / 1000:.2f}s")
    print(f"Avg time/page: {total_time / len(pages_results):.0f}ms")
    print(f"{'=' * 60}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Dolphin OCR Benchmark")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Dolphin model")
    parser.add_argument("--input_path", required=True, help="Path to input image/TIFF")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--max_pages", type=int, help="Max pages to process")
    args = parser.parse_args()

    run_benchmark(args.model_path, args.input_path, args.output_dir, args.max_pages)


if __name__ == "__main__":
    main()
