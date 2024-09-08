########################################################################################################################
# 本代码内所有注释（版权提示、风险提示）等内容均不得删除。
#
# 版权所有 © 2024 徐少卿 Bradley.xsq@gmail.com. 保留所有权利.
#
# 本代码受Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License（CC BY-NC-ND 4.0）保护。
#
# 根据此许可，您可以：
# - 分享：以任何媒介或格式复制、分发该代码的副本。
#
# 需遵守以下条款：
# - 署名：您必须给予适当的署名，提供该许可的链接，并注明是否进行了修改。
# - 非商业用途：您不得将本代码用于商业用途。
# - 禁止演绎：如果您对本代码进行再混合、转换或在其基础上进行创作，您不得分发修改后的代码。
# - 没有额外的限制：您不得施加法律术语或技术措施，限制他人依许可允许的行为。
#
# 详细信息请参阅 https://creativecommons.org/licenses/by-nc-nd/4.0/

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

# Block 2_2 function
# Function to process each file in block2_2
def process_file_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
    global model_block2_2, tokenizer_block2_2  # 声明模型和tokenizer为全局变量

    # Load the model and tokenizer
    tokenizer_block2_2 = AutoTokenizer.from_pretrained(
        model_path_block2_2,
        trust_remote_code=True,
        encode_special_tokens=True
    )

    model_block2_2 = AutoModel.from_pretrained(
        model_path_block2_2,
        trust_remote_code=True,
        device_map="auto").eval()

    class StopOnTokensBlock2_2(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = model_block2_2.config.eos_token_id
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # Ensure the output folder exists
    os.makedirs(output_folder_block2_2, exist_ok=True)

    output_log_block2_2 = ""  # Accumulate output log

    try:
        # Process each file in the input folder
        for filename_block2_2 in os.listdir(input_folder_block2_2):
            file_path_block2_2 = os.path.join(input_folder_block2_2, filename_block2_2)
            if os.path.isfile(file_path_block2_2):
                output_log_block2_2 += f"已开始处理： {filename_block2_2}，请稍后...\n"
                yield gr.update(value=output_log_block2_2)  # Update the output log with processing info

                with open(file_path_block2_2, 'r', encoding='utf-8') as file_block2_2:
                    content_block2_2 = file_block2_2.read()

                command_block2_2 = f"请将以下文本整理成问答形式（严格按照 Q：    A：    不加序号，）后给我，问题里禁止使用简称、代号等(例如学院得写成完整的学院名字，课程得写明课程名字): {content_block2_2}"

                messages_block2_2 = [{"role": "user", "content": command_block2_2}]

                model_inputs_block2_2 = tokenizer_block2_2.apply_chat_template(
                    messages_block2_2,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt"
                ).to(model_block2_2.device)

                streamer_block2_2 = TextIteratorStreamer(
                    tokenizer=tokenizer_block2_2,
                    timeout=60,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                stop_block2_2 = StopOnTokensBlock2_2()
                generate_kwargs_block2_2 = {
                    "input_ids": model_inputs_block2_2,
                    "streamer": streamer_block2_2,
                    "max_new_tokens": 8192,
                    "do_sample": True,
                    "top_p": 0.8,
                    "temperature": 0.6,
                    "stopping_criteria": StoppingCriteriaList([stop_block2_2]),
                    "repetition_penalty": 1.2,
                    "eos_token_id": model_block2_2.config.eos_token_id,
                }

                t_block2_2 = Thread(target=model_block2_2.generate, kwargs=generate_kwargs_block2_2)
                t_block2_2.start()

                output_text_block2_2 = ""
                for new_token_block2_2 in streamer_block2_2:
                    if new_token_block2_2:
                        output_text_block2_2 += new_token_block2_2

                # Save the output text to the processed_texts folder
                output_filename_block2_2 = os.path.join(output_folder_block2_2, f"final_{os.path.basename(filename_block2_2)}")
                with open(output_filename_block2_2, 'w', encoding='utf-8') as output_file_block2_2:
                    output_file_block2_2.write(output_text_block2_2.strip())

                output_log_block2_2 += f" {filename_block2_2} 已处理并存储至 {output_filename_block2_2}\n"
                yield gr.update(value=output_log_block2_2)  # Update the output log with completion info
            else:
                output_log_block2_2 += f"{file_path_block2_2} 不是目标文件.\n"
                yield gr.update(value=output_log_block2_2)

    finally:
        # 清理模型、数据和GPU资源
        unload_model_block2_2()
        output_log_block2_2 += f"此进程已完成，GPU已释放。请开始下一个 BLOCK 任务。\n"
        yield gr.update(value=output_log_block2_2)
    # 卸载模型并释放GPU内存

def unload_model_block2_2():
    global model_block2_2, tokenizer_block2_2
    if 'model_block2_2' in globals():
        del model_block2_2
    if 'tokenizer_block2_2' in globals():
        del tokenizer_block2_2
    gc.collect()  # 垃圾回收
    torch.cuda.empty_cache()  # 清理缓存
    torch.cuda.synchronize()  # 确保所有CUDA操作完成

# Gradio Interface for block2_2
def gradio_interface_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
    for output_block2_2 in process_file_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
        yield output_block2_2

# Gradio Blocks
with gr.Blocks(title="UMTCDS") as demo:
    gr.Markdown("### Block-2.2 处理文本数据(本地模型)")
    with gr.Group():
        with gr.Row():
            model_path_block2_2 = gr.Textbox(label="模型路径", placeholder="样例：/root/autodl-tmp/glm-4-9b-chat")

        with gr.Row():
            input_folder_block2_2 = gr.Textbox(label="文件输入路径 ", placeholder="请输入 txt所在路径，可使用绝对路径或相对路径")
            output_folder_block2_2 = gr.Textbox(label="文件输出路径", placeholder="请输入 txt输出路径，可使用绝对路径或相对路径")
            execute_button_block2_2 = gr.Button("运行 BLOCK2.2", variant="primary")

        with gr.Row():
            output_textbox_block2_2 = gr.Textbox(label="实时日志", lines=5)
            execute_button_block2_2.click(fn=gradio_interface_block2_2,inputs=[model_path_block2_2, input_folder_block2_2, output_folder_block2_2],outputs=output_textbox_block2_2)

# Launch Gradio demo
    demo.launch(server_name="0.0.0.0", server_port=6006)
