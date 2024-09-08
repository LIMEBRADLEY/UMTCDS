########################################################################################################################
# All comments (copyright notices, risk warnings) and other content in this code must not be deleted.
#
# Copyright © 2024 Shaoqing XU Bradley.xsq@gmail.com . All rights reserved.
#
# This code is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public
# License (CC BY-NC-ND 4.0).
#
# Under this license, you are free to:
# - Share: Copy and redistribute the material in any medium or format.
#
# Under the following terms: - Attribution: You must give appropriate credit, provide a link to the license,
# and indicate if changes were made. - NonCommercial: You may not use the material for commercial purposes. -
# NoDerivatives: If you remix, transform, or build upon the material, you may not distribute the modified material. -
# No additional restrictions: You may not apply legal terms or technological measures that legally restrict others
# from doing anything the license permits.
#
# For more information, please visit https://creativecommons.org/licenses/by-nc-nd/4.0/
########################################################################################################################
import gradio as gr
import random
import requests
from bs4 import BeautifulSoup
import time
import yaml
from urllib.parse import urlparse
from threading import Thread
from tqdm import tqdm
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel
import gc
import json
import re
import os
import ruamel.yaml as yaml
import torch
import subprocess
from ruamel.yaml import YAML
import os
from pathlib import Path
from threading import Thread
from typing import Union
import pandas as pd
import gradio as gr
import torch
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)
from zhipuai import ZhipuAI
import threading

# Block 2_1 function
# Function to process each file in block2_1
def process_file_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
    global model_block2_1, tokenizer_block2_1  # 声明模型和tokenizer为全局变量
    # Load the model and tokenizer
    tokenizer_block2_1 = AutoTokenizer.from_pretrained(
        model_path_block2_1,
        trust_remote_code=True,
        encode_special_tokens=True
    )

    model_block2_1 = AutoModel.from_pretrained(
        model_path_block2_1,
        trust_remote_code=True,
        device_map="auto").eval()

    class StopOnTokensBlock2_1(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = model_block2_1.config.eos_token_id
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # Ensure the output folder exists
    os.makedirs(output_folder_block2_1, exist_ok=True)

    output_log_block2_1 = ""  # Accumulate output log

    try:
        # Process each file in the input folder
        for filename_block2_1 in os.listdir(input_folder_block2_1):
            file_path_block2_1 = os.path.join(input_folder_block2_1, filename_block2_1)
            if os.path.isfile(file_path_block2_1):
                output_log_block2_1 += f"Processing: {filename_block2_1}, please wait...\n"
                yield gr.update(value=output_log_block2_1)  # Update the output log with processing info

                with open(file_path_block2_1, 'r', encoding='utf-8') as file_block2_1:
                    content_block2_1 = file_block2_1.read()

                command_block2_1 = f"Please send me the following text in paragraphs :{content_block2_1}"

                messages_block2_1 = [{"role": "user", "content": command_block2_1}]

                model_inputs_block2_1 = tokenizer_block2_1.apply_chat_template(
                    messages_block2_1,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt"
                ).to(model_block2_1.device)

                streamer_block2_1 = TextIteratorStreamer(
                    tokenizer=tokenizer_block2_1,
                    timeout=60,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                stop_block2_1 = StopOnTokensBlock2_1()
                generate_kwargs_block2_1 = {
                    "input_ids": model_inputs_block2_1,
                    "streamer": streamer_block2_1,
                    "max_new_tokens": 8192,
                    "do_sample": True,
                    "top_p": 0.8,
                    "temperature": 0.6,
                    "stopping_criteria": StoppingCriteriaList([stop_block2_1]),
                    "repetition_penalty": 1.2,
                    "eos_token_id": model_block2_1.config.eos_token_id,
                }

                t_block2_1 = Thread(target=model_block2_1.generate, kwargs=generate_kwargs_block2_1)
                t_block2_1.start()

                output_text_block2_1 = ""
                for new_token_block2_1 in streamer_block2_1:
                    if new_token_block2_1:
                        output_text_block2_1 += new_token_block2_1

                # Save the output text to the processed_texts folder
                output_filename_block2_1 = os.path.join(output_folder_block2_1, f"processed_{os.path.basename(filename_block2_1)}")
                with open(output_filename_block2_1, 'w', encoding='utf-8') as output_file_block2_1:
                    output_file_block2_1.write(output_text_block2_1.strip())

                output_log_block2_1 += f" {filename_block2_1} processed and stored in {output_filename_block2_1}\n"
                yield gr.update(value=output_log_block2_1)  # Update the output log with completion info
            else:
                output_log_block2_1 += f"{file_path_block2_1} is not a target file.\n"
                yield gr.update(value=output_log_block2_1)
    finally:
        # 清理模型、数据和GPU资源
        unload_model_block2_1()
        output_log_block2_1 += f"This process has completed and the GPU has been released. Please start the next BLOCK task.\n"
        yield gr.update(value=output_log_block2_1)  # Yield GPU release message

def unload_model_block2_1():
    global model_block2_1, tokenizer_block2_1
    if 'model_block2_1' in globals():
        del model_block2_1
    if 'tokenizer_block2_1' in globals():
        del tokenizer_block2_1
    gc.collect()  # 垃圾回收
    torch.cuda.empty_cache()  # 清理缓存
    torch.cuda.synchronize()  # 确保所有CUDA操作完成


# Gradio Interface for block2_1
def gradio_interface_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
    for output_block2_1 in process_file_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
        yield output_block2_1

# Gradio Blocks
with gr.Blocks(title="UMTCDS") as demo:
    gr.Markdown("### Block-2.1 Summarizing Text Data (Local Model)")
    with gr.Group():
        with gr.Row():
            model_path_block2_1 = gr.Textbox(label="Model Path", placeholder="Example：/root/autodl-tmp/glm-4-9b-chat")

        with gr.Row():
            input_folder_block2_1 = gr.Textbox(label="Input Path ", placeholder="Please enter the path where txt is located. You can use an absolute path or a relative path.")
            output_folder_block2_1 = gr.Textbox(label="Output Path", placeholder="Please enter the txt output path.")
            execute_button_block2_1 = gr.Button("Start BLOCK2.1", variant="primary")

        with gr.Row():
            output_textbox_block2_1 = gr.Textbox(label="Real-time log", lines=5)
            execute_button_block2_1.click(fn=gradio_interface_block2_1,inputs=[model_path_block2_1, input_folder_block2_1, output_folder_block2_1],outputs=output_textbox_block2_1)

# Launch Gradio demo
    demo.launch(server_name="0.0.0.0", server_port=6006)
