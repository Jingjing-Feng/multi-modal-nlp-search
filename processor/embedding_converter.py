import cv2
import numpy as np
import pytesseract
import torch
import whisper
from pdf2image import convert_from_path
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)


# Load the models

# Image captioning model
BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Audio model
WHISPER_MODEL = whisper.load_model("base")

# Video model
LLAVA_PROCESSOR = LlavaProcessor.from_pretrained(
    "llava-hf/llava-interleave-qwen-0.5b-hf"
)
LLAVA_MODEL = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-interleave-qwen-0.5b-hf", torch_dtype=torch.float16
).to("cuda")


def convert_to_pure_text(file_path: str) -> str:
    """
    Convert the text file to embedding
    """
    with open(file_path) as f:
        pure_text = f.read()

    return pure_text


def convert_image_to_text(file_path: str) -> str:
    """
    Convert the image to text by image captioning and OCR and then to embedding
    """
    image = Image.open(file_path).convert("RGB")

    # OCR
    ocr_result = pytesseract.image_to_string(image)

    # Image captioning
    image_tensor = BLIP_PROCESSOR(image, return_tensors="pt")
    output_tensor = BLIP_MODEL.generate(**image_tensor)
    image_caption = BLIP_PROCESSOR.decode(output_tensor[0], skip_special_tokens=True)

    full_text = image_caption + "\n" + ocr_result

    return full_text


def convert_pdf_to_text(file_path: str) -> str:
    """
    Convert the pdf file to text and then to embedding
    """
    pdf_pages = convert_from_path(file_path)
    pdf_page_texts = [
        pytesseract.image_to_string(page.convert("RGB")) for page in pdf_pages
    ]
    pdf_full_text = "\n".join(pdf_page_texts)
    return pdf_full_text


def convert_audio_to_text(file_path: str) -> str:
    """
    Convert the audio file to text and then to embedding
    """
    audio_transcript = WHISPER_MODEL.transcribe(file_path)["text"]
    return audio_transcript


def _sample_frames(path, num_frames):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    frames = []

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        if i % interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert(
                "RGB"
            )
            frames.append(pil_img)

    video.release()
    return frames[:num_frames]


def convert_video_to_text(file_path: str) -> str:
    """
    Convert the video file to text and then to embedding
    """
    videos = _sample_frames(file_path, 6)

    user_prompt = "Describe this video"
    toks = "<image>" * len(videos)
    prompt = f"<|im_start|>user{toks}\n{user_prompt}<|im_end|><|im_start|>assistant"

    inputs = LLAVA_PROCESSOR(text=prompt, images=videos, return_tensors="pt").to(
        LLAVA_MODEL.device
    )
    output = LLAVA_MODEL.generate(**inputs, max_new_tokens=100, do_sample=False)
    output_text = LLAVA_PROCESSOR.decode(output[0], skip_special_tokens=True)[
        len(user_prompt) :
    ]
    return output_text