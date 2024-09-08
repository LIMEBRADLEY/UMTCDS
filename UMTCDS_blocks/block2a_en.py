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
import os
import threading
from zhipuai import ZhipuAI

# 用于控制线程停止的事件
stop_event_block2a = threading.Event()


# Block 2a function
def block2a_process_files(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
    global stop_event_block2a
    stop_event_block2a.clear()  # 开始前清除停止事件
    logs = []  # 初始化日志列表

    try:
        block2a_client = ZhipuAI(api_key=block2a_api_key)
        # 进行一次请求，确保API key是有效的
        block2a_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "这是测试消息，用于校验API key。"}]
        )
        log = "API key is valid and ZhipuAI client initialized.\n"
        logs.append(log)
        yield ''.join(logs)
    except Exception as e:
        # 如果捕获到身份验证错误，则终止操作
        log = f"API key is invalid or cannot initialize ZhipuAI client:{e}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    # 确保保存路径存在
    try:
        os.makedirs(block2a_output_folder_path, exist_ok=True)
        log = f"Output folder created in: {block2a_output_folder_path}\n"
        logs.append(log)
        yield ''.join(logs)
    except Exception as e:
        log = f"Unable to create output folder: {e}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    # 检查输入文件夹是否存在并且包含txt文件
    if not os.path.exists(block2a_folder_path):
        log = f"The input folder path does not exist: {block2a_folder_path}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    block2a_txt_files_found = False

    # 遍历文件夹下的所有子文件夹和txt文件
    for root, dirs, files in os.walk(block2a_folder_path):
        if stop_event_block2a.is_set():
            log = "The process has been interrupted.\n"
            logs.append(log)
            yield ''.join(logs)
            return

        # 对文件进行排序，以确保按顺序处理
        files.sort(key=lambda f: int(os.path.splitext(f)[0].split('_')[0]))

        for file in files:
            if stop_event_block2a.is_set():
                log = "The process has been interrupted.\n"
                logs.append(log)
                yield ''.join(logs)
                return

            if file.endswith(".txt"):
                block2a_txt_files_found = True
                try:
                    # 获取txt文件的完整路径
                    block2a_file_path = os.path.join(root, file)

                    # 读取txt文件内容
                    with open(block2a_file_path, 'r', encoding='utf-8') as f:
                        block2a_file_content = f.read()

                    # 调用GLM-4生成回答
                    block2a_response = block2a_client.chat.completions.create(
                        model="glm-4-flash",  # 使用GLM-4模型
                        messages=[
                            {"role": "user",
                             "content": "Please clean the following text and remove all irrelevant content. Then, convert the cleaned content into question-answer pairs. Each question-answer pair should be accurate and relevant to the original text content."},
                            {"role": "assistant",
                             "content": "Sure, send me your text and I'll clean it up and generate question-answer pairs, as (Q: A:)."},
                            {"role": "user",
                             "content": f"This is my data, please help me organize it and return only the correct answers:{block2a_file_content}"}
                        ],
                    )

                    if not block2a_response.choices or not block2a_response.choices[0].message:
                        log = f"Processing of file {file} failed: No valid API response received\n"
                        logs.append(log)
                        yield ''.join(logs)
                        continue

                    # 获取GLM-4的回答内容并转为字符串
                    block2a_glm_response_content = block2a_response.choices[0].message.content

                    # 生成保存文件的名字
                    block2a_output_file_name = f"processed_{file}"
                    block2a_output_file_path = os.path.join(block2a_output_folder_path, block2a_output_file_name)

                    # 保存GLM-4的回答到新文件
                    with open(block2a_output_file_path, 'w', encoding='utf-8') as block2a_output_file:
                        block2a_output_file.write(block2a_glm_response_content)

                    log = f"Files processed and saved to:{block2a_output_file_path}\n"
                    logs.append(log)
                    yield ''.join(logs)
                except Exception as e:
                    log = f"An error occurred while processing file {file}: {e}\n"
                    logs.append(log)
                    yield ''.join(logs)

    if not block2a_txt_files_found:
        log = "No .txt files found, please check the input folder.\n"
        logs.append(log)
        yield ''.join(logs)
    else:
        log = "All files processed successfully.\n"
        logs.append(log)
        yield ''.join(logs)


# 用于处理线程的启动和停止
def run_block2a(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
    logs = ""  # 重置日志内容
    for result in block2a_process_files(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
        logs = result  # 只保留本次结果的日志
        yield logs


def stop_block2a():
    stop_event_block2a.set()  # 设置事件为 True，表示停止
    return "BLOCK2A task is stopping, please wait..."


# Gradio interface setup
with gr.Blocks(title="UMTCDS") as demo:
    # Block 2A
    gr.Markdown("### Block2A-Processing text data (API)")
    gr.Markdown("Please provide ZHIPU API KEY, input folder path and output folder path. The program will process the text and save the result.")
    with gr.Group():
        with gr.Row():
            block2a_api_key_input = gr.Textbox(label="API Key", placeholder="Please enter Zhipu API KEY")
            block2a_folder_path_input = gr.Textbox(label="Input Path", placeholder="Please enter the input folder path")
            block2a_output_folder_path_input = gr.Textbox(label="Output Path", placeholder="Please enter the output folder path")

        with gr.Row():
            block2a_run_button = gr.Button("Start BLOCK2A", variant="primary")
            block2a_stop_button = gr.Button("Stop BLOCK2A", variant="secondary")

        block2a_log_output = gr.Textbox(label="Real-time log", lines=5, interactive=False)

        block2a_run_button.click(fn=run_block2a, inputs=[block2a_api_key_input, block2a_folder_path_input,
                                                         block2a_output_folder_path_input], outputs=block2a_log_output)
        block2a_stop_button.click(fn=stop_block2a, outputs=block2a_log_output)

    demo.launch()
