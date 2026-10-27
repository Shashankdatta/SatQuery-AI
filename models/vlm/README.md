# Vision-Language Model (VLM) Module

## Planned Architecture: Qwen2.5-VL-3B-Instruct

This directory will host the Vision-Language Model pipeline for remote-sensing question answering.

### Model Details
- **Base Architecture**: Qwen2.5-VL-3B-Instruct
- **Task**: Visual Question Answering (VQA) over high-resolution satellite and aerial imagery.
- **Fine-Tuning Strategy**: LoRA / QLoRA parameter-efficient fine-tuning on remote sensing datasets (e.g., RSVQA, RSIVQA, Sydney, UCMerced).

### Modules to be added
1. `loader.py`: Model checkpoint loading with 4-bit/8-bit quantization support via `bitsandbytes` or `vLLM`.
2. `prompt_formatter.py`: Specialized system prompts for geospatial context, coordinate systems, and grounding.
3. `finetune_lora.py`: Training script for PEFT/LoRA adaptation on domain imagery.
