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
import subprocess

# Block 6 function
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
    )
    return model, tokenizer


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(history, prompt, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    for idx, (user_msg, model_msg) in enumerate(history):
        if prompt and idx == 0:
            continue
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
        yield history


# 强制GPU复位功能
# def reset_gpu():
#     try:
#         result = subprocess.run(["nvidia-smi", "--gpu-reset"], check=True)
#         print("GPU reset successful.")
#         return "GPU 已成功复位。"
#     except subprocess.CalledProcessError as e:
#         print(f"GPU reset failed: {e}")
#         return f"GPU复位失败: {e}"

# 卸载模型并释放GPU内存的函数
def unload_model():
    global model, tokenizer
    if 'model' in globals():
        del model  # 删除模型
    if 'tokenizer' in globals():
        del tokenizer  # 删除tokenizer

    gc.collect()  # 运行垃圾回收器
    torch.cuda.empty_cache()  # 清除缓存
    torch.cuda.synchronize()  # 等待所有CUDA操作完成

    # 强制复位GPU
    # reset_message = reset_gpu()

    return "模型已卸载，GPU内存已释放。恭喜您完成此次微调进程，期待您的下一次使用！"


# Block 6 end


# Gradio界面
with gr.Blocks() as demo:
    # Block 6
    gr.Markdown("### Block 6 推理并校验")
    model_path_input = gr.Textbox(label="微调后权重路径", placeholder="请输入： ./output/checkpoint-xxx")


    def load_model_on_click(model_path):
        global model, tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path, trust_remote_code=True)
        return f"微调后权重： {model_path} 已推理并加载完成。"


    with gr.Row():
        load_model_btn = gr.Button("开始 推理", variant="primary")
        unload_model_btn = gr.Button("卸载模型", variant="secondary")

    unload_model_btn.click(unload_model, outputs=model_path_input)

    chatbot = gr.Chatbot()

    with gr.Row():
        # 左侧用户输入部分
        with gr.Column(scale=3):
            user_input = gr.Textbox(show_label=False, placeholder="请输入：", lines=10, container=False)
            submitBtn = gr.Button("发送", variant="primary")

        # 中间 Prompt 设置部分
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(show_label=False, placeholder="Prompt", lines=10, container=False)
            pBtn = gr.Button("Set Prompt")

        # 右侧模型控制部分
        with gr.Column(scale=1):
            emptyBtn = gr.Button("清除对话历史")
            max_length = gr.Slider(0, 32768, value=256, step=1.0, label="最大长度", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    def user(query, history):
        return "", history + [[query, ""]]


    def set_prompt(prompt_text):
        return [[prompt_text, "成功设置prompt"]]


    pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)

    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, prompt_input, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

    load_model_btn.click(load_model_on_click, inputs=[model_path_input], outputs=[model_path_input])

    demo.launch(server_name="0.0.0.0", server_port=6006)
