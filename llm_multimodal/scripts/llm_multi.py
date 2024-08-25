import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig, AutoProcessor
import gradio as gr
from threading import Thread
from PIL import Image
import whisper
import requests
whisper_model = whisper.load_model("base")
API_KEY = os.environ.get("API_KEY", "<>")
# Install flash-attn if not already installed

# Function to transcribe audio using Whisper
def transcribe_audio_whisper(audio):
    result = whisper_model.transcribe(audio)
    print(result["text"])
    # breakpoint()
    return result["text"]

def transcribe_audio_saaras(audio):
    # URL and headers
    url = "https://api.sarvam.ai/speech-to-text-translate"
    headers = {
        'api-subscription-key': API_KEY
    }

    files = {
        'file': ('', open(audio, 'rb'), 'audio/wav')
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files)

    return (response.text.lstrip('{"transcript":"').split('","language_code')[0])

# Vision model setup
models = {
    "microsoft/Phi-3.5-vision-instruct": AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True, torch_dtype="auto", _attn_implementation="flash_attention_2").cuda().eval()
}

processors = {
    "microsoft/Phi-3.5-vision-instruct": AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)
}

user_prompt = '\n'
assistant_prompt = '\n'
prompt_suffix = "\n"

# Vision model tab function
def stream_vision(image, audio_input=None, model_id="microsoft/Phi-3.5-vision-instruct", whisp="Whisper"):
    model = models[model_id]
    processor = processors[model_id]

    # Convert audio input to text using Whisper
    if audio_input:
        if whisp == "Whisper":
            text_input = transcribe_audio_whisper(audio_input)
        else:
            text_input = transcribe_audio_saaras(audio_input)
        print(f"Audio input: {text_input}")
    else:
        text_input = ""

    model = models[model_id]
    processor = processors[model_id]

    # Prepare the image list and corresponding tags
    images = [Image.fromarray(image).convert("RGB")]
    placeholder = "<|image_1|>\n"  # Using the image tag as per the example

    # Construct the prompt with the image tag and the user's text input
    if text_input:
        prompt_content = placeholder + text_input
    else:
        prompt_content = placeholder

    messages = [
        {"role": "user", "content": prompt_content},
    ]

    # Apply the chat template to the messages
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the inputs with the processor
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    # Generation parameters
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate the response
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove input tokens from the generated response
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

    # Decode the generated output
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response, text_input

# Gradio app with two tabs
with gr.Blocks() as demo:
    gr.Markdown(f"<h1><center>MultiModal VoiceBot</center></h1>")
    with gr.Row():
        input_img = gr.Image(label="Input Picture")
    with gr.Row():
        model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value="microsoft/Phi-3.5-vision-instruct")
    with gr.Row():
        whisp = gr.Dropdown(choices=['Whisper', 'Saaras'], label="Translation Model", value="Whisper")            
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Question (Speak)")
    with gr.Row():
        text_ves = gr.Textbox(label="Speech to Text")
    with gr.Row():
        submit_btn = gr.Button(value="Submit")
    with gr.Row():
        output_text = gr.Textbox(label="Output Text")

    submit_btn.click(stream_vision, [input_img, audio_input, model_selector, whisp], [output_text, text_ves])

# Launch the combined app
demo.launch(share=True)