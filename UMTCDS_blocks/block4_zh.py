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
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, urljoin
import threading
from ruamel.yaml import YAML

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

    # Use the recommended settings for ruamel.yaml
    yaml = YAML(typ='unsafe', pure=True)

    # Write the configuration to the YAML file
    with open(block4_config_output_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file)

    return f"配置文件已经生成在： {block4_config_output_path}。请开始下一进程。"

# Gradio interface setup
with gr.Blocks(title="UMTCDS") as demo:
    # Block 4 with Redesigned Layout
    gr.Markdown("### Block-4: 生成配置文件", elem_id="step4-title")

    # Data Configuration Section
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

    # Training Arguments Section
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

    # Logging & Evaluation Section
    with gr.Group():
        gr.Markdown("#### 日志和评估")
        with gr.Row():
            log_level_input_4 = gr.Dropdown(label="日志等级", choices=["debug", "info", "warn", "error"], value="info")
            logging_strategy_input_4 = gr.Dropdown(label="日志策略", choices=["steps", "epoch"], value="steps")
            logging_steps_input_4 = gr.Slider(label="日志步数", minimum=10, maximum=500, step=10, value=10)
        with gr.Row():
            per_device_eval_batch_size_input_4 = gr.Slider(label="评估批次大小", minimum=1, maximum=64, step=1,
                                                           value=16)
            eval_strategy_input_4 = gr.Dropdown(label="评估策略", choices=["steps", "epoch"], value="steps")
            eval_steps_input_4 = gr.Slider(label="评估步数", minimum=100, maximum=10000, step=500, value=1000)
            predict_with_generate_input_4 = gr.Checkbox(label="使用生成进行预测", value=True)
            max_new_tokens_input_4 = gr.Slider(label="最大新tokens数量", minimum=128, maximum=1024, step=64, value=512)

    # PEFT Configuration Section
    with gr.Group():
        gr.Markdown("#### PEFT 配置")
        with gr.Row():
            peft_type_input_4 = gr.Dropdown(label="PEFT 类别", choices=["PREFIX_TUNING", "ADAPTER", "BITFIT"],
                                            value="PREFIX_TUNING")
            task_type_input_4 = gr.Dropdown(label="任务列别", choices=["CAUSAL_LM", "SEQ2SEQ_LM"], value="CAUSAL_LM")
        with gr.Row():
            num_virtual_tokens_input_4 = gr.Slider(label="虚拟tokens数量", minimum=128, maximum=1024, step=64,
                                                   value=512)
            num_attention_heads_input_4 = gr.Slider(label="注意头数量", minimum=1, maximum=16, step=1, value=2)
            token_dim_input_4 = gr.Slider(label="Token 维度", minimum=128, maximum=1024, step=64, value=256)

    # Output Configuration Section
    with gr.Group():
        gr.Markdown("#### 输出配置")
        config_output_path_input_4 = gr.Textbox(label="样例： ./Step4_output_path/config.yaml")
        block4_button = gr.Button("Start Block4", variant="primary")

    block4_output = gr.Textbox(label="实时日志", lines=1, max_lines=4, interactive=False)

    # Bind function to button for Block 4
    block4_button.click(generate_config_yaml,
                        inputs=[
                            train_file_input_4, val_file_input_4, test_file_input_4, num_proc_input_4, combine_input_4,
                            max_input_length_input_4, max_output_length_input_4, output_dir_input_4, max_steps_input_4,
                            learning_rate_input_4, per_device_train_batch_size_input_4, dataloader_num_workers_input_4,
                            remove_unused_columns_input_4, save_strategy_input_4, save_steps_input_4, log_level_input_4,
                            logging_strategy_input_4, logging_steps_input_4, per_device_eval_batch_size_input_4,
                            eval_strategy_input_4, eval_steps_input_4, predict_with_generate_input_4,
                            max_new_tokens_input_4,
                            peft_type_input_4, task_type_input_4, num_virtual_tokens_input_4,
                            num_attention_heads_input_4,
                            token_dim_input_4, config_output_path_input_4
                        ],
                        outputs=block4_output
                        )

    demo.launch()
