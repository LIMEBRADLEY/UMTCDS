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
# 风险提示
legal_notice = """
<div style="text-align: center;">
    <h1>法律免责声明和开发者责任声明</h1>
</div>

<h2>法律合规提示</h2>
通过使用UMTCDS，您确认并同意遵守与网页爬取和数据提取相关的您所在司法管辖区的适用法律和规定，包括但不限于：

**中华人民共和国（PRC）:**
   - 遵守《中华人民共和国网络安全法》，该法律对网络运营商规定了确保网络产品和服务安全的义务。
   - 遵守《个人信息保护法》（PIPL），该法律规范个人信息的处理，要求在收集、使用或传输个人数据前需获得个人明确的同意。
    <blockquote>
        <strong>中華人民共和國香港特別行政區（香港）：</strong>
        <ul>
        <li>遵守《個人資料（私隱）條例》（PDPO），該條例規定了個人資料私隱的保護，包括數據收集、保存、準確性和安全性的要求。</li>
        <li>用戶必須確保任何數據處理活動符合PDPO的原則，如需要從數據主體處獲取明確和顯著的同意以及在使用個人資料時的透明度。</li>
        </ul>

        <strong>中華人民共和國澳門特別行政區（澳門）：</strong>
        <ul>
        <li>遵守《個人資料保護法》（法案第8/2005號），該法規定了在澳門收集、處理和存儲個人資料的要求。用戶負責獲取適當的同意，並確保處理的數據的安全性和保密性。</li>
        <li>用戶還必須遵守有關跨境數據傳輸的規定以及根據澳門法律賦予數據主體的權利。</li>
        </ul>

        <strong>中華人民共和國 台灣省：</strong>
        <ul>
        <li>遵守《個人資料保護法》，該法律規範在颱灣收集、處理和使用個人資料的行爲。機構必須確保個人資料合法收集並且符合收集目的的使用。</li>
        <li>用戶負責在收集、處理或使用個人資料前，獲取個人的適當同意，並確保個人資料的安全和透明使用。</li>
        </ul></blockquote>

<h2>开发者责任：</h2>

UMTCDS的开发者明确表示，对于以违反任何适用法律或法规的方式使用本软件，不承担任何责任。用户应承担全部责任，确保其对本软件的使用符合其管辖范围内的所有相关法律要求。

即使已被告知发生此类损害的可能性，开发者也不应对因使用或无法使用本软件而产生的或与之相关的任何直接、间接、偶然、特殊或后果性损害承担责任。

<h2>用户须知：</h2>

在使用此软件提取数据之前，您有责任查看并遵守目标网站的服务条款、隐私政策和法律要求。未经授权的数据抓取、爬取或提取可能会导致法律后果。在使用此软件之前，请务必确保您已获得适当的同意或法律授权。"""


def handle_agreement(agree):
    if agree:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return "You have declined to comply with the legal requirements. The application will stop.", gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False)


css_style = """footer{display:flex;justify-content:center;align-items:center;padding:5px 0;font-size:14px;}footer .gr-footer-content{display:none;}"""

########################################################################################################################


from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin
import matplotlib.pyplot as plt
import random
import requests
from bs4 import BeautifulSoup
import time
import yaml
from urllib.parse import urlparse
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel
import gc
import json
import re
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import os
from pathlib import Path
from threading import Thread
from typing import Union
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

# Function Code Start
stop_event_block1 = threading.Event()


def clean_text_block1(soup):
    for element in soup.find_all(
            ['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript', 'iframe', 'embed', 'object', 'form',
             'input', 'button']):
        element.extract()

    for element in soup.find_all(
            lambda tag: (tag.name in ['div', 'span', 'section', 'article', 'aside', 'ul', 'li']) and (
                    'ad' in tag.get('class', []) or
                    'ads' in tag.get('class', []) or
                    'advertisement' in tag.get('class', []) or
                    'sponsored' in tag.get('class', []) or
                    'footer' in tag.get('class', []) or
                    'header' in tag.get('class', []) or
                    'sidebar' in tag.get('class', []) or
                    'popup' in tag.get('class', []) or
                    'modal' in tag.get('class', []) or
                    'banner' in tag.get('class', []) or
                    'cookie' in tag.get('class', []) or
                    'widget' in tag.get('class', []) or
                    'social' in tag.get('class', []) or
                    'related' in tag.get('class', []) or
                    'tracking' in tag.get('class', []) or
                    'comment' in tag.get('class', []) or
                    'breadcrumbs' in tag.get('class', []) or
                    'outbrain' in tag.get('class', []) or
                    'taboola' in tag.get('class', []) or
                    'yahoo' in tag.get('class', []) or
                    'newsletter' in tag.get('class', []) or
                    'contact' in tag.get('class', []) or
                    'promo' in tag.get('class', []) or
                    'announcement' in tag.get('class', []) or
                    'alert' in tag.get('class', []) or
                    'nav' in tag.get('class', []) or
                    'subscribe' in tag.get('class', []) or
                    'subscribe' in tag.get('id', []) or
                    'scroll' in tag.get('class', []) or
                    'carousel' in tag.get('class', []) or
                    'newsletter' in tag.get('class', []) or
                    'share' in tag.get('class', []) or
                    'vote' in tag.get('class', []) or
                    'vote' in tag.get('id', []) or
                    'email' in tag.get('class', []) or
                    'email' in tag.get('id', []) or
                    'map' in tag.get('class', []) or
                    'map' in tag.get('id', []) or
                    'download' in tag.get('class', []) or
                    'download' in tag.get('id', [])
            )):
        element.extract()

    for element in soup.find_all(lambda tag: 'cookie' in tag.get('id', '') or 'cookie' in tag.get('class', []) or
                                             'modal' in tag.get('id', '') or 'modal' in tag.get('class', []) or
                                             'popup' in tag.get('id', '') or 'popup' in tag.get('class', []) or
                                             'subscribe' in tag.get('id', '') or 'subscribe' in tag.get('class', [])):
        element.extract()

    text = soup.get_text(separator=' ').replace('\xa0', ' ').replace('&nbsp;', ' ')
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return ' '.join(cleaned_lines)


def is_valid_url_block1(href, base_domain):
    parsed_href = urlparse(href)
    href_domain = parsed_href.netloc
    if parsed_href.scheme not in ["http", "https", ""]:
        return False
    if href_domain == "":
        return True

    return href_domain.endswith(base_domain)


def extract_text_from_html_block1(url, max_pages, output_dir):
    stop_event_block1.clear()
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=5)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    visited_urls = set()
    downloaded_titles = set()
    page_count = 0
    file_counter = 1

    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc.split('.')[-2] + '.' + parsed_url.netloc.split('.')[-1]

    output_dir = os.path.abspath(output_dir)
    content_dir = os.path.join(output_dir, "extracted_texts_block1")
    os.makedirs(content_dir, exist_ok=True)

    def visit_page(current_url, depth=0, prefix=''):
        nonlocal page_count, file_counter

        if stop_event_block1.is_set():
            print("程序终止！请重新运行此BLOCK！")
            return

        if current_url in visited_urls or depth >= max_pages or page_count >= max_pages:
            return
        visited_urls.add(current_url)

        try:
            response = session.get(current_url, verify=True, timeout=10)
            response.encoding = 'utf-8'
        except requests.exceptions.Timeout:
            yield f"请求超时，跳过 {current_url}"
            return
        except requests.exceptions.SSLError:
            yield f"SSL 证书错误，跳过 {current_url}"
            return
        except requests.exceptions.RequestException as e:
            yield f"请求错误: {e}"
            return
        except Exception as e:
            yield f"发生异常: {e}"
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        cleaned_text = clean_text_block1(soup)

        title = soup.title.string if soup.title else f"page_{depth + 1}"
        title = title.replace("/", "-").replace("\\", "-") if title else f"untitled_page_{depth + 1}"

        if prefix:
            title = f"{prefix}_{title}"

        if title in downloaded_titles:
            yield f"跳过 {title} (已下载)"
            return

        file_name = f"{file_counter:03d}_{title}.txt"
        output_file = os.path.join(content_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        downloaded_titles.add(title)
        file_counter += 1
        page_count += 1
        yield f"[{page_count}/{max_pages}] 文本成功提取并保存到 {output_file}"

        if depth == 0:
            links = soup.find_all('a', href=True)
            for link in links:
                if stop_event_block1.is_set():
                    print("程序终止！请重新运行此BLOCK！")
                    return
                href = link['href']
                link_text = link.get_text(strip=True).replace("/", "-").replace("\\", "-")
                if is_valid_url_block1(href, base_domain):
                    if not href.startswith('http'):
                        href = urljoin(current_url, href)
                    if href not in visited_urls:
                        yield from visit_page(href, depth + 1, prefix=link_text)

    yield from visit_page(url)
    yield "目标 URL 页面及其子页面已全部保存。请开始下一个 BLOCK 任务。\n"


def validate_url_block1(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def validate_output_dir_block1(output_dir):
    return os.path.exists(output_dir) and os.access(output_dir, os.W_OK)


def run_block1(url, max_pages, output_dir):
    if not validate_url_block1(url):
        yield "您输入的 #URL 无效，请重新输入。"
        return

    if not validate_output_dir_block1(output_dir):
        yield "您输入的 #输出位置 无效，请重新输入。"
        return

    log = ""
    for result in extract_text_from_html_block1(url, max_pages, output_dir):
        if stop_event_block1.is_set():
            log += "\nBLOCK1 已经被手动停止，您可以重新运行本 BLOCK1 任务。"
            yield log
            break
        log += "\n" + result
        yield log


def stop_extraction_block1():
    stop_event_block1.set()  # 设置事件为 True，表示停止
    print("stop_event_block1 set to True")
    return "PLEASE WAIT 20 seconds"


stop_event_block2a = threading.Event()


def block2a_process_files(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
    global stop_event_block2a
    stop_event_block2a.clear()
    logs = []

    try:
        block2a_client = ZhipuAI(api_key=block2a_api_key)
        block2a_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "这是测试消息，用于校验API key。"}]
        )
        log = "API key is valid and ZhipuAI client initialized.\n"
        logs.append(log)
        yield ''.join(logs)
    except Exception as e:
        # 如果捕获到身份验证错误，则终止操作
        log = f"API key无效或无法初始化ZhipuAI客户端: {e}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    try:
        os.makedirs(block2a_output_folder_path, exist_ok=True)
        log = f"输出文件夹创建于: {block2a_output_folder_path}\n"
        logs.append(log)
        yield ''.join(logs)
    except Exception as e:
        log = f"无法创建输出文件夹: {e}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    if not os.path.exists(block2a_folder_path):
        log = f"输入文件夹路径不存在: {block2a_folder_path}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    block2a_txt_files_found = False

    for root, dirs, files in os.walk(block2a_folder_path):
        if stop_event_block2a.is_set():
            log = "进程已被中断。\n"
            logs.append(log)
            yield ''.join(logs)
            return

        files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[0]))

        for file in files:
            if stop_event_block2a.is_set():
                log = "进程已被中断。\n"
                logs.append(log)
                yield ''.join(logs)
                return

            if file.endswith(".txt"):
                block2a_txt_files_found = True
                try:
                    block2a_file_path = os.path.join(root, file)

                    with open(block2a_file_path, 'r', encoding='utf-8') as f:
                        block2a_file_content = f.read()

                    block2a_response = block2a_client.chat.completions.create(
                        model="glm-4-flash",  # 使用GLM-4模型
                        messages=[
                            {"role": "user",
                             "content": "请你对以下文本进行清理，去除所有无关内容。然后，将清理后的内容转换为问答对，每个问答对应该准确且与原始文本内容相关。"},
                            {"role": "assistant",
                             "content": "当然，请将您的文本发送给我，我将进行清理并生成问答对，按照（Q： A：）。"},
                            {"role": "user",
                             "content": f"这是我的数据，请帮我整理并只返回给我问答对：{block2a_file_content}"}
                        ],
                    )

                    if not block2a_response.choices or not block2a_response.choices[0].message:
                        log = f"文件 {file} 的处理失败：未收到有效的API响应\n"
                        logs.append(log)
                        yield ''.join(logs)
                        continue

                    block2a_glm_response_content = block2a_response.choices[0].message.content

                    block2a_output_file_name = f"processed_{file}"
                    block2a_output_file_path = os.path.join(block2a_output_folder_path, block2a_output_file_name)

                    with open(block2a_output_file_path, 'w', encoding='utf-8') as block2a_output_file:
                        block2a_output_file.write(block2a_glm_response_content)

                    log = f"文件已处理并保存：{block2a_output_file_path}\n"
                    logs.append(log)
                    yield ''.join(logs)
                except Exception as e:
                    log = f"处理文件 {file} 时发生错误: {e}\n"
                    logs.append(log)
                    yield ''.join(logs)

    if not block2a_txt_files_found:
        log = "未找到任何.txt文件，请检查输入文件夹。\n"
        logs.append(log)
        yield ''.join(logs)
    else:
        log = "所有文件已成功处理。请开始下一进程。\n"
        logs.append(log)
        yield ''.join(logs)


def run_block2a(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
    logs = ""
    for result in block2a_process_files(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
        logs = result
        yield logs


def stop_block2a():
    stop_event_block2a.set()
    return "BLOCK2A任务正在停止，请稍候..."


def process_file_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
    # Load the model and tokenizer
    global model_block2_1, tokenizer_block2_1
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

    os.makedirs(output_folder_block2_1, exist_ok=True)

    output_log_block2_1 = ""

    try:

        for filename_block2_1 in os.listdir(input_folder_block2_1):
            file_path_block2_1 = os.path.join(input_folder_block2_1, filename_block2_1)
            if os.path.isfile(file_path_block2_1):
                output_log_block2_1 += f"已开始处理： {filename_block2_1}，请稍后...\n"
                yield gr.update(value=output_log_block2_1)

                with open(file_path_block2_1, 'r', encoding='utf-8') as file_block2_1:
                    content_block2_1 = file_block2_1.read()

                command_block2_1 = f"请将以下文本整理成段后给我: {content_block2_1}"

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

                output_filename_block2_1 = os.path.join(output_folder_block2_1,
                                                        f"processed_{os.path.basename(filename_block2_1)}")
                with open(output_filename_block2_1, 'w', encoding='utf-8') as output_file_block2_1:
                    output_file_block2_1.write(output_text_block2_1.strip())

                output_log_block2_1 += f" {filename_block2_1} 已处理并存储至 {output_filename_block2_1}\n"
                yield gr.update(value=output_log_block2_1)  # Update the output log with completion info
            else:
                output_log_block2_1 += f"{file_path_block2_1} 不是目标文件.\n"
                yield gr.update(value=output_log_block2_1)
    finally:
        unload_model_block2_1()
        output_log_block2_1 += f"此进程已完成，GPU已释放。请开始下一个 BLOCK 任务。\n"
        yield gr.update(value=output_log_block2_1)


def unload_model_block2_1():
    global model_block2_1, tokenizer_block2_1
    if 'model_block2_1' in globals():
        del model_block2_1
    if 'tokenizer_block2_1' in globals():
        del tokenizer_block2_1

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gradio_interface_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
    for output_block2_1 in process_file_block2_1(model_path_block2_1, input_folder_block2_1, output_folder_block2_1):
        yield output_block2_1


def process_file_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
    global model_block2_2, tokenizer_block2_2

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

    os.makedirs(output_folder_block2_2, exist_ok=True)

    output_log_block2_2 = ""

    try:
        for filename_block2_2 in os.listdir(input_folder_block2_2):
            file_path_block2_2 = os.path.join(input_folder_block2_2, filename_block2_2)
            if os.path.isfile(file_path_block2_2):
                output_log_block2_2 += f"已开始处理： {filename_block2_2}，请稍后...\n"
                yield gr.update(value=output_log_block2_2)

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

                output_filename_block2_2 = os.path.join(output_folder_block2_2,
                                                        f"final_{os.path.basename(filename_block2_2)}")
                with open(output_filename_block2_2, 'w', encoding='utf-8') as output_file_block2_2:
                    output_file_block2_2.write(output_text_block2_2.strip())

                output_log_block2_2 += f" {filename_block2_2} 已处理并存储至 {output_filename_block2_2}\n"
                yield gr.update(value=output_log_block2_2)
            else:
                output_log_block2_2 += f"{file_path_block2_2} 不是目标文件.\n"
                yield gr.update(value=output_log_block2_2)
    finally:
        unload_model_block2_2()
        output_log_block2_2 += f"此进程已完成，GPU已释放。请开始下一个 BLOCK 任务。\n"
        yield gr.update(value=output_log_block2_2)


def unload_model_block2_2():
    global model_block2_2, tokenizer_block2_2
    if 'model_block2_2' in globals():
        del model_block2_2
    if 'tokenizer_block2_2' in globals():
        del tokenizer_block2_2
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gradio_interface_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
    for output_block2_2 in process_file_block2_2(model_path_block2_2, input_folder_block2_2, output_folder_block2_2):
        yield output_block2_2


def block3_remove_quotes(text):
    return re.sub(r'[“”""]', '', text)


def block3_validate_inputs(input_folder, base_folder, filenames):
    if not os.path.exists(input_folder):
        raise ValueError(f"输入目录： {input_folder} 不存在！")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    for filename in filenames:
        if not filename.endswith('.jsonl'):
            raise ValueError(f"文件名非法： {filename}. 必须以 '.jsonl'结尾")


def block3_process_file(file_path):
    all_contents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            all_contents.extend(content.split("\n\n"))
    except Exception:
        pass
    return all_contents


def block3_extract_qa_pairs_parallel(input_folder):
    """使用多线程并行处理多个文件"""
    with ThreadPoolExecutor() as executor:
        file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                      os.path.isfile(os.path.join(input_folder, f))]
        results = list(executor.map(block3_process_file, file_paths))
    return [qa_pair for result in results for qa_pair in result]  # 合并结果


def block3_convert_to_jsonl(input_folder, base_folder, train_filename, val_filename, test_filename):
    block3_validate_inputs(input_folder, base_folder, [train_filename, val_filename, test_filename])

    output_files = {
        "train": os.path.join(base_folder, train_filename),
        "val": os.path.join(base_folder, val_filename),
        "test": os.path.join(base_folder, test_filename)
    }

    all_contents = block3_extract_qa_pairs_parallel(input_folder)
    random.shuffle(all_contents)
    total_count = len(all_contents)
    train_split = int(total_count * 0.8)
    val_split = train_split + int(total_count * 0.1)

    for i, qa_pair in enumerate(all_contents):
        if "Q：" in qa_pair and "A：" in qa_pair:
            question = qa_pair.split("Q：")[1].split("A：")[0].strip()
            answer = qa_pair.split("A：")[1].strip()
            question = block3_remove_quotes(question)
            json_content = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }

            if i < train_split:
                with open(output_files["train"], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(json_content, ensure_ascii=False) + "\n")
            elif i < val_split:
                with open(output_files["val"], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(json_content, ensure_ascii=False) + "\n")
            else:
                with open(output_files["test"], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(json_content, ensure_ascii=False) + "\n")

    return "所有问答对都已转换并分为训练、验证和测试集三个jsonl文件。请开始下一个 BLOCK 任务。"


def run_block3(input_folder, base_folder, train_filename, val_filename, test_filename):
    log = block3_convert_to_jsonl(input_folder, base_folder, train_filename, val_filename, test_filename)
    yield log


def generate_config_yaml(block4_train_file, block4_val_file, block4_test_file, block4_num_proc, block4_combine,
                         block4_max_input_length, block4_max_output_length, block4_output_dir, block4_max_steps,
                         block4_learning_rate, block4_per_device_train_batch_size, block4_dataloader_num_workers,
                         block4_remove_unused_columns, block4_save_strategy, block4_save_steps, block4_log_level,
                         block4_logging_strategy, block4_logging_steps, block4_per_device_eval_batch_size,
                         block4_eval_strategy, block4_eval_steps, block4_predict_with_generate, block4_max_new_tokens,
                         block4_peft_type, block4_task_type, block4_num_virtual_tokens, block4_num_attention_heads,
                         block4_token_dim, block4_config_output_path):
    config = {
        "data_config": {
            "train_file": block4_train_file,
            "val_file": block4_val_file,
            "test_file": block4_test_file,
            "num_proc": block4_num_proc
        },
        "combine": block4_combine,
        "max_input_length": block4_max_input_length,
        "max_output_length": block4_max_output_length,
        "training_args": {
            "output_dir": block4_output_dir,
            "max_steps": block4_max_steps,
            "learning_rate": block4_learning_rate,
            "per_device_train_batch_size": block4_per_device_train_batch_size,
            "dataloader_num_workers": block4_dataloader_num_workers,
            "remove_unused_columns": block4_remove_unused_columns,
            "save_strategy": block4_save_strategy,
            "save_steps": block4_save_steps,
            "log_level": block4_log_level,
            "logging_strategy": block4_logging_strategy,
            "logging_steps": block4_logging_steps,
            "per_device_eval_batch_size": block4_per_device_eval_batch_size,
            "eval_strategy": block4_eval_strategy,
            "eval_steps": block4_eval_steps,
            "predict_with_generate": block4_predict_with_generate,
            "generation_config": {
                "max_new_tokens": block4_max_new_tokens
            }
        },
        "peft_config": {
            "peft_type": block4_peft_type,
            "task_type": block4_task_type,
            "num_virtual_tokens": block4_num_virtual_tokens,
            "num_attention_heads": block4_num_attention_heads,
            "token_dim": block4_token_dim
        }
    }

    yaml = YAML(typ='unsafe', pure=True)

    with open(block4_config_output_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file)

    return f"配置文件已经生成在： {block4_config_output_path}。请开始下一进程。"


def block5_run_finetuning(block5_data_dir, block5_model_dir, block5_config_file, block5_auto_resume_from_checkpoint):
    command = f"python UMTCDS_ft.py {block5_data_dir} {block5_model_dir} {block5_config_file} {block5_auto_resume_from_checkpoint}"

    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          bufsize=1) as process:
        block5_full_output = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                block5_full_output += output.strip() + "\n"
                yield block5_full_output

    yield block5_full_output + "微调进程已经结束，请在Block6中进行推理对话。"


def block5_load_config(block5_config_file):
    try:
        yaml_loader = yaml.YAML(typ='safe', pure=True)
        with open(block5_config_file, 'r') as f:
            block5_config = yaml_loader.load(f)

        block5_output_dir = block5_config["training_args"]["output_dir"]
        block5_save_steps = block5_config["training_args"]["save_steps"]
        block5_max_steps = block5_config["training_args"]["max_steps"]
        return block5_output_dir, block5_save_steps, block5_max_steps
    except yaml.YAMLError as e:
        print(f"YAML decode error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None, None, None


def block5_monitor_checkpoints(block5_output_dir, block5_save_steps, block5_max_steps):
    block5_processed_steps = set()
    block5_loss_data = []
    block5_step_data = []

    while True:
        block5_checkpoint_found = False

        for step in range(block5_save_steps, block5_max_steps + block5_save_steps, block5_save_steps):
            block5_checkpoint_dir = os.path.join(block5_output_dir, f"checkpoint-{step}")
            block5_trainer_state_file = os.path.join(block5_checkpoint_dir, "trainer_state.json")

            if os.path.exists(block5_trainer_state_file):
                block5_checkpoint_found = True
                with open(block5_trainer_state_file, 'r') as f:
                    block5_trainer_state = json.load(f)
                    for log in block5_trainer_state["log_history"]:
                        if log["step"] not in block5_processed_steps:
                            block5_loss_data.append(log["loss"])
                            block5_step_data.append(log["step"])
                            block5_processed_steps.add(log["step"])

        if block5_checkpoint_found:
            if len(block5_step_data) > 0:
                yield block5_step_data, block5_loss_data
            else:
                yield None, None
        else:
            print("No checkpoints found yet. Waiting...")
            yield None, None

        time.sleep(4)


def block5_plot_loss(block5_step_data, block5_loss_data):
    plt.figure(figsize=(10, 6))
    plt.plot(block5_step_data, block5_loss_data, marker='o', color='b', label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("block5_loss_vs_steps.png")
    return "block5_loss_vs_steps.png"


def block5_start_monitoring(block5_config_file):
    block5_output_dir, block5_save_steps, block5_max_steps = block5_load_config(block5_config_file)

    if not block5_output_dir:
        return "Error loading config file."

    for block5_step_data, block5_loss_data in block5_monitor_checkpoints(block5_output_dir, block5_save_steps,
                                                                         block5_max_steps):
        if block5_step_data and block5_loss_data:
            yield block5_plot_loss(block5_step_data, block5_loss_data)
        else:
            yield None


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


def unload_model():
    global model, tokenizer
    if 'model' in globals():
        del model
    if 'tokenizer' in globals():
        del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return "模型已卸载，GPU内存已释放。恭喜您完成此次微调进程，期待您的下一次使用！"


def load_model_on_click(model_path):
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, trust_remote_code=True)
    return f"微调后权重： {model_path} 已推理并加载完成。"


def user(query, history):
    return "", history + [[query, ""]]


def set_prompt(prompt_text):
    return [[prompt_text, "成功设置prompt"]]


# All Function Code End
########################################################################################################################


# Gradio UI Start
with gr.Blocks(title="UMTCDS", css=css_style) as UMTCDS:
    warning_text = gr.Markdown(legal_notice, visible=True)
    agree_button = gr.Button("我同意", visible=True)
    disagree_button = gr.Button("我不同意", visible=True)
    output_text = gr.Markdown(visible=False)
    main_interface = gr.Column(visible=False)
    agree_button.click(handle_agreement, inputs=[gr.State(True)],
                       outputs=[warning_text, main_interface, agree_button, disagree_button])
    disagree_button.click(handle_agreement, inputs=[gr.State(False)],
                          outputs=[output_text, warning_text, agree_button, disagree_button])

    with main_interface:
        gr.Markdown("""
                <script>
                    document.addEventListener("DOMContentLoaded", function() {
                        const outputbox = document.querySelector(".outputbox");
                        outputbox.scrollTop = outputbox.scrollHeight;  // 保持滚动条在底部
                    });

                    function keepScrollPosition() {
                        window.scrollTo(0, document.body.scrollHeight);
                    }

                    setInterval(keepScrollPosition, 100);  // 每100毫秒保持一次滚动位置
                </script>
                """)

        gr.Markdown("""
                <div style="text-align: center;">
                    <h1>A Universal Method for Fine-Tuning Campus Dialogue Systems: A Case Study Using GLM4-9B</h1>
                    <h3>Author: Xu Shaoqing   E-mail: bradley.xsq@gmail.com</h3>
                    <h3>University: Macao University of Science and Technology  </h3>

                </div>
                """)

        gr.Markdown("### Block1-从目标URL中获取初始数据 ")
        gr.Markdown("请输入 #URL，代码会自动递归访问URL的第一层子URL，每个URL都将被转换为txt格式文档存储在你的 #输出位置 ")
        with gr.Group():
            with gr.Row():
                url_input_block1 = gr.Textbox(label="URL",
                                              placeholder="样例：https://fie.must.edu.mo/index.html?locale=zh_CN",
                                              elem_id="url-input-block1")
                output_dir_input_block1 = gr.Textbox(label="输出位置 此处创建项目根目录",
                                                     value="./Your_project_non-core_files",
                                                     elem_id="output-dir-input-block1")
                max_pages_input_block1 = gr.Slider(label="URL访问次数", minimum=1, maximum=100, step=1, value=50,
                                                   elem_id="max-pages-input-block1")

            with gr.Row():
                submit_button_block1 = gr.Button("运行 BLOCK1 ", variant="primary", elem_id="submit-button-block1")
                stop_button_block1 = gr.Button("终止 BLOCK1", variant="secondary", elem_id="stop-button-block1")

            output_block1 = gr.Textbox(label="实时日志", lines=5, interactive=False, elem_id="output-box-block1")

        submit_button_block1.click(run_block1,
                                   inputs=[url_input_block1, max_pages_input_block1, output_dir_input_block1],
                                   outputs=output_block1)
        stop_button_block1.click(stop_extraction_block1, outputs=output_block1)

        gr.Markdown("### Block2A-处理文本数据 (API)。")
        gr.Markdown("请提供API KEY、输入文件夹路径和输出文件夹路径。程序将处理文本并保存结果。")
        with gr.Group():
            with gr.Row():
                block2a_api_key_input = gr.Textbox(label="API Key", placeholder="请输入智谱API KEY")
                block2a_folder_path_input = gr.Textbox(label="输入位置", placeholder="请输入输入文件夹路径")
                block2a_output_folder_path_input = gr.Textbox(label="输出位置", placeholder="请输入输出文件夹路径")

            with gr.Row():
                block2a_run_button = gr.Button("运行 BLOCK2A", variant="primary")
                block2a_stop_button = gr.Button("停止 BLOCK2A", variant="secondary")

            block2a_log_output = gr.Textbox(label="实时日志", lines=5, interactive=False)

            block2a_run_button.click(fn=run_block2a, inputs=[block2a_api_key_input, block2a_folder_path_input,
                                                             block2a_output_folder_path_input],
                                     outputs=block2a_log_output)
            block2a_stop_button.click(fn=stop_block2a, outputs=block2a_log_output)

        gr.Markdown("### Block2.1-总结文本数据(本地模型)")
        gr.Markdown("使用本地GLM-4-9B-Chat 进行文本整理，请先从魔塔社区拉去模型权重文件")
        with gr.Group():
            with gr.Row():
                model_path_block2_1 = gr.Textbox(label="模型路径", value="/root/autodl-tmp/glm-4-9b-chat")

            with gr.Row():
                input_folder_block2_1 = gr.Textbox(label="文件输入路径",
                                                   placeholder="请输入 txt所在路径，可使用绝对路径或相对路径")
                output_folder_block2_1 = gr.Textbox(label="文件输出路径",
                                                    placeholder="请输入 txt输出路径，可使用绝对路径或相对路径")
                execute_button_block2_1 = gr.Button("运行 BLOCK2.1", variant="primary")

            with gr.Row():
                output_textbox_block2_1 = gr.Textbox(label="实时日志", lines=5)
                execute_button_block2_1.click(fn=gradio_interface_block2_1,
                                              inputs=[model_path_block2_1, input_folder_block2_1,
                                                      output_folder_block2_1],
                                              outputs=output_textbox_block2_1)

        gr.Markdown("### Block2.2-处理文本数据(本地模型)")
        gr.Markdown("使用本地GLM-4-9B-Chat 进行问答对转换，请先从魔塔社区拉去模型权重文件")

        with gr.Group():
            with gr.Row():
                model_path_block2_2 = gr.Textbox(label="模型路径", placeholder="样例：/root/autodl-tmp/glm-4-9b-chat")

            with gr.Row():
                input_folder_block2_2 = gr.Textbox(label="文件输入路径 ",
                                                   placeholder="请输入 txt所在路径，可使用绝对路径或相对路径")
                output_folder_block2_2 = gr.Textbox(label="文件输出路径",
                                                    placeholder="请输入 txt输出路径，可使用绝对路径或相对路径")
                execute_button_block2_2 = gr.Button("运行 BLOCK2.2", variant="primary")

            with gr.Row():
                output_textbox_block2_2 = gr.Textbox(label="实时日志", lines=5)
                execute_button_block2_2.click(fn=gradio_interface_block2_2,
                                              inputs=[model_path_block2_2, input_folder_block2_2,
                                                      output_folder_block2_2],
                                              outputs=output_textbox_block2_2)

        gr.Markdown("### Block3-生成JSONL数据集")
        gr.Markdown("该部分将批量转换所有txt文件为三个JSONL数据集文件。")
        with gr.Group():
            with gr.Row():
                block3_input_path = gr.Textbox(label="输入路径", placeholder="请输入待转换的txt文件所在路径",
                                               elem_id="block3-input-path")
                block3_base_folder = gr.Textbox(label="Output Path", placeholder="请输入 数据集存放位置",
                                                elem_id="block3-base-folder")
            with gr.Row():
                block3_train_filename = gr.Textbox(label="训练集文件名",
                                                   placeholder="请输入训练集文件名称 (e.g., train.jsonl)",
                                                   elem_id="block3-train-filename")
                block3_val_filename = gr.Textbox(label="验证集文件名",
                                                 placeholder="请输入验证集文件名称(e.g., val.jsonl)",
                                                 elem_id="block3-val-filename")
                block3_test_filename = gr.Textbox(label="测试集文件名",
                                                  placeholder="请输入测试集文件名称 (e.g., test.jsonl)",
                                                  elem_id="block3-test-filename")
            with gr.Row():
                block3_button = gr.Button("开始 Block3", variant="primary", elem_id="block3-button")
            block3_output = gr.Textbox(label="JSONL Conversion and Split Result", lines=5, interactive=False,
                                       elem_id="block3-output-box")

        block3_button.click(run_block3,
                            inputs=[block3_input_path, block3_base_folder, block3_train_filename, block3_val_filename,
                                    block3_test_filename], outputs=block3_output)

        gr.Markdown("### Block4-生成配置文件", elem_id="step4-title")

        with gr.Group():
            gr.Markdown("#### 数据配置")
            with gr.Row():
                train_file_input_4 = gr.Textbox(label="数据集（训练）的名称", value="train.jsonl")
                val_file_input_4 = gr.Textbox(label="数据集（验证）的名称", value="val.jsonl")
                test_file_input_4 = gr.Textbox(label="数据集（测试）的名称", value="test.jsonl")
                num_proc_input_4 = gr.Slider(label="进程数量", minimum=1, maximum=16, step=1, value=1)
            with gr.Row():
                combine_input_4 = gr.Checkbox(label="合并数据文件", value=True)
                max_input_length_input_4 = gr.Slider(label="最大输入长度", minimum=64, maximum=512, step=32, value=128)
                max_output_length_input_4 = gr.Slider(label="最大输出长度", minimum=128, maximum=1024, step=64,
                                                      value=512)
        with gr.Group():
            gr.Markdown("#### 训练参数")
            with gr.Row():
                output_dir_input_4 = gr.Textbox(label="权重输出位置", value="./output")
                max_steps_input_4 = gr.Slider(label="训练总步数", minimum=100, maximum=100000, step=100, value=5000)
                learning_rate_input_4 = gr.Number(label="学习率", value=3e-3, precision=4)
                per_device_train_batch_size_input_4 = gr.Slider(label="训练批次", minimum=1, maximum=64, step=1,
                                                                value=4)
            with gr.Row():
                dataloader_num_workers_input_4 = gr.Slider(label="数据加载器", minimum=1, maximum=32, step=1,
                                                           value=16)
                remove_unused_columns_input_4 = gr.Checkbox(label="删除未使用的列", value=False)
                save_strategy_input_4 = gr.Dropdown(label="保存策略", choices=["steps", "epoch"], value="steps")
                save_steps_input_4 = gr.Slider(label="保存步数", minimum=50, maximum=10000, step=50, value=500)

        with gr.Group():
            gr.Markdown("#### 日志和评估")
            with gr.Row():
                log_level_input_4 = gr.Dropdown(label="日志等级", choices=["debug", "info", "warn", "error"],
                                                value="info")
                logging_strategy_input_4 = gr.Dropdown(label="日志策略", choices=["steps", "epoch"], value="steps")
                logging_steps_input_4 = gr.Slider(label="日志步数", minimum=10, maximum=500, step=10, value=10)
            with gr.Row():
                per_device_eval_batch_size_input_4 = gr.Slider(label="评估批次大小", minimum=1, maximum=64, step=1,
                                                               value=16)
                eval_strategy_input_4 = gr.Dropdown(label="评估策略", choices=["steps", "epoch"], value="steps")
                eval_steps_input_4 = gr.Slider(label="评估步数", minimum=100, maximum=10000, step=500, value=1000)
                predict_with_generate_input_4 = gr.Checkbox(label="使用生成进行预测", value=True)
                max_new_tokens_input_4 = gr.Slider(label="最大新tokens数量", minimum=128, maximum=1024, step=64,
                                                   value=512)

        with gr.Group():
            gr.Markdown("#### PEFT 配置")
            with gr.Row():
                peft_type_input_4 = gr.Dropdown(label="PEFT 类别", choices=["PREFIX_TUNING", "ADAPTER", "BITFIT"],
                                                value="PREFIX_TUNING")
                task_type_input_4 = gr.Dropdown(label="任务列别", choices=["CAUSAL_LM", "SEQ2SEQ_LM"],
                                                value="CAUSAL_LM")
            with gr.Row():
                num_virtual_tokens_input_4 = gr.Slider(label="虚拟tokens数量", minimum=128, maximum=1024, step=64,
                                                       value=512)
                num_attention_heads_input_4 = gr.Slider(label="注意头数量", minimum=1, maximum=16, step=1, value=2)
                token_dim_input_4 = gr.Slider(label="Token 维度", minimum=128, maximum=1024, step=64, value=256)

        with gr.Group():
            gr.Markdown("#### 输出配置")
            config_output_path_input_4 = gr.Textbox(label="样例： ./Step4_output_path/config.yaml")
            block4_button = gr.Button("Start Block4", variant="primary")

        block4_output = gr.Textbox(label="实时日志", lines=1, max_lines=4, interactive=False)

        block4_button.click(generate_config_yaml,
                            inputs=[
                                train_file_input_4, val_file_input_4, test_file_input_4, num_proc_input_4,
                                combine_input_4,
                                max_input_length_input_4, max_output_length_input_4, output_dir_input_4,
                                max_steps_input_4,
                                learning_rate_input_4, per_device_train_batch_size_input_4,
                                dataloader_num_workers_input_4,
                                remove_unused_columns_input_4, save_strategy_input_4, save_steps_input_4,
                                log_level_input_4,
                                logging_strategy_input_4, logging_steps_input_4, per_device_eval_batch_size_input_4,
                                eval_strategy_input_4, eval_steps_input_4, predict_with_generate_input_4,
                                max_new_tokens_input_4,
                                peft_type_input_4, task_type_input_4, num_virtual_tokens_input_4,
                                num_attention_heads_input_4,
                                token_dim_input_4, config_output_path_input_4
                            ],
                            outputs=block4_output
                            )

        gr.Markdown("### Block5-微调进程")
        gr.Markdown("使用本地GLM-4-9B-Chat 进行模型微调，微调流程采用官方代码")
        with gr.Group():
            with gr.Row():
                block5_data_dir = gr.Textbox(label="数据集路径", placeholder="请输入数据集所在路径")
                block5_model_dir = gr.Textbox(label="模型路径", value="/root/autodl-tmp/glm-4-9b-chat")
                block5_config_file = gr.Textbox(label="配置文件",
                                                placeholder="样例： ./Your_project_non-core_files/config.yaml")
                block5_auto_resume_from_checkpoint = gr.Textbox(label="从检查点自动恢复", value="",
                                                                placeholder="初始微调无需输入任何字段")

        with gr.Row():
            block5_output = gr.Textbox(label="微调 实时日志",
                                       placeholder="加载模型需要消耗15s左右，生成初始图像需要30s左右", lines=20,
                                       max_lines=20)
            block5_graph_output = gr.Image(label="Loss vs Steps Graph")

        with gr.Row():
            block5_run_button = gr.Button("开始微调", variant="primary")
            block5_generate_graph_button = gr.Button("开始图像生成", variant="primary")

            block5_run_button.click(block5_run_finetuning, [block5_data_dir, block5_model_dir, block5_config_file,
                                                            block5_auto_resume_from_checkpoint], block5_output)
            block5_generate_graph_button.click(block5_start_monitoring, [block5_config_file], block5_graph_output)

        gr.Markdown("### Block6-推理并校验")
        gr.Markdown("点击开始推理后，显示loaded后，方可与模型进行对话。")
        model_path_input = gr.Textbox(label="微调后权重路径", placeholder="请输入： ./output/checkpoint-xxx")

        with gr.Row():
            load_model_btn = gr.Button("开始 推理", variant="primary")
            unload_model_btn = gr.Button("卸载模型", variant="secondary")

        unload_model_btn.click(unload_model, outputs=model_path_input)

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=3):
                user_input = gr.Textbox(show_label=False, placeholder="请输入：", lines=10, container=False)
                submitBtn = gr.Button("发送", variant="primary")

            with gr.Column(scale=2):
                prompt_input = gr.Textbox(show_label=False, placeholder="Prompt", lines=10, container=False)
                pBtn = gr.Button("Set Prompt")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("清除对话历史")
                max_length = gr.Slider(0, 32768, value=256, step=1.0, label="最大长度", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

        pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)

        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            predict, [chatbot, prompt_input, max_length, top_p, temperature], chatbot
        )
        emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

        load_model_btn.click(load_model_on_click, inputs=[model_path_input], outputs=[model_path_input])

# Gradio UI Code End
########################################################################################################################
        # 任何使用者禁止移出一下声明内容
        gr.HTML("""<footer><p>
                    © 2024 UMTCDS - Author: <strong>Shaoqing XU</strong> - Email: <strong>Bradley.xsq@gmail.com</strong> All rights reserved.
                    This project is licensed under the <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank">
                    CC BY-NC-ND 4.0</a> License.
                </p></footer>""")

########################################################################################################################

    UMTCDS.launch(server_name="0.0.0.0", server_port=6006)
