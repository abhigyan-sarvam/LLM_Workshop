# Saaras-Llama VoiceBot
>Colab link: https://colab.research.google.com/drive/1pJzUYE0PifRHD3S3DPr4V6sIb-OI_z1R?usp=sharing

## Run Ollama using Docker
docker run -d --gpus=all -v ollama:/root/.ollama -p 8034:11434 --name ollama ollama/ollama

## Download llama3.1 and gemma2 models
docker exec -it ollama ollama run llama3.1
docker exec -it ollama ollama run gemma2

# MultiModal VoiceBot

## Installation and Setup
> Install pytorch from official repo - https://pytorch.org/get-started/locally/
> Install other requirements - pip install -r requirements.txt
> Install flash attention - pip install flash-attn --no-build-isolation

## Run gradio app
python scripts/llm_multi.py
