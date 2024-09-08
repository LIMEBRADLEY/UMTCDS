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
import os
import json
import time
import subprocess
import matplotlib.pyplot as plt
import gradio as gr
import yaml


# 启动微调进程
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
                yield block5_full_output  # 实时输出日志信息

    yield block5_full_output + "Fine-tuning process completed."


# 读取配置文件获取 output_dir
def block5_load_config(block5_config_file):
    try:
        with open(block5_config_file, 'r') as f:
            block5_config = yaml.safe_load(f)

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


# 实时监控output_dir目录，读取新的checkpoint文件夹中的trainer_state.json
def block5_monitor_checkpoints(block5_output_dir, block5_save_steps, block5_max_steps):
    block5_processed_steps = set()  # 用于追踪已处理的step
    block5_loss_data = []
    block5_step_data = []

    # 循环监控目录
    while True:
        block5_checkpoint_found = False

        # 遍历output_dir中的所有checkpoint文件夹
        for step in range(block5_save_steps, block5_max_steps + block5_save_steps, block5_save_steps):
            block5_checkpoint_dir = os.path.join(block5_output_dir, f"checkpoint-{step}")
            block5_trainer_state_file = os.path.join(block5_checkpoint_dir, "trainer_state.json")

            # 如果文件夹和trainer_state.json存在且step未被处理
            if os.path.exists(block5_trainer_state_file):
                block5_checkpoint_found = True
                with open(block5_trainer_state_file, 'r') as f:
                    block5_trainer_state = json.load(f)
                    for log in block5_trainer_state["log_history"]:
                        if log["step"] not in block5_processed_steps:
                            block5_loss_data.append(log["loss"])
                            block5_step_data.append(log["step"])
                            block5_processed_steps.add(log["step"])  # 记录已处理的step

        if block5_checkpoint_found:
            if len(block5_step_data) > 0:  # 如果有数据
                yield block5_step_data, block5_loss_data
            else:
                yield None, None
        else:
            print("No checkpoints found yet. Waiting...")
            yield None, None

        time.sleep(4)  # 等待10秒后重新检查目录


# 绘制实时更新的loss-step图像
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


# Gradio界面回调函数，用于开始监控并生成图像
def block5_start_monitoring(block5_config_file):
    block5_output_dir, block5_save_steps, block5_max_steps = block5_load_config(block5_config_file)

    if not block5_output_dir:
        return "Error loading config file."

    for block5_step_data, block5_loss_data in block5_monitor_checkpoints(block5_output_dir, block5_save_steps,
                                                                         block5_max_steps):
        if block5_step_data and block5_loss_data:
            yield block5_plot_loss(block5_step_data, block5_loss_data)  # 实时生成图像
        else:
            yield None  # 如果没有新数据，则不更新图像


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("### Block5 Start the fine-tuning process")
    with gr.Group():
        with gr.Row():
            block5_data_dir = gr.Textbox(label="Data Directory", placeholder="Enter Step4_output_path/")
            block5_model_dir = gr.Textbox(label="Model Directory", value="/root/autodl-tmp/glm-4-9b-chat")
            block5_config_file = gr.Textbox(label="Configuration File",
                                            placeholder="Enter ./Step1_output_path/config.yaml")
            block5_auto_resume_from_checkpoint = gr.Textbox(label="Auto Resume From Checkpoint", value="",
                                                            placeholder="Leave empty or provide checkpoint number")

    with gr.Row():
        block5_output = gr.Textbox(label="Fine-tune Process Result",
                                   placeholder="The output of the process will appear here...", lines=20, max_lines=20)
        block5_graph_output = gr.Image(label="Loss vs Steps Graph")

    with gr.Row():
        block5_run_button = gr.Button("Run Fine-Tuning", variant="primary")
        block5_generate_graph_button = gr.Button("Start Monitoring", variant="primary")

        block5_run_button.click(block5_run_finetuning, [block5_data_dir, block5_model_dir, block5_config_file,
                                                        block5_auto_resume_from_checkpoint], block5_output)
        block5_generate_graph_button.click(block5_start_monitoring, [block5_config_file], block5_graph_output)

    # 启动界面
    demo.launch()
