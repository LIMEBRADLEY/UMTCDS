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

    return f"Config file saved at {block4_config_output_path}. Please start the next BLOCK task.\n"

# Gradio interface setup
with gr.Blocks(title="UMTCDS") as demo:
    # Block 4 with Redesigned Layout
    gr.Markdown("### Block-4: Generate Config YAML File", elem_id="step4-title")

    # Data Configuration Section
    with gr.Group():
        gr.Markdown("#### Data Configuration")
        with gr.Row():
            train_file_input_4 = gr.Textbox(label="Train File", value="train.jsonl")
            val_file_input_4 = gr.Textbox(label="Validation File", value="val.jsonl")
            test_file_input_4 = gr.Textbox(label="Test File", value="test.jsonl")
            num_proc_input_4 = gr.Slider(label="Number of Processes", minimum=1, maximum=16, step=1, value=1)
        with gr.Row():
            combine_input_4 = gr.Checkbox(label="Combine Data Files", value=True)
            max_input_length_input_4 = gr.Slider(label="Max Input Length", minimum=64, maximum=512, step=32, value=128)
            max_output_length_input_4 = gr.Slider(label="Max Output Length", minimum=128, maximum=1024, step=64,
                                                  value=512)

    # Training Arguments Section
    with gr.Group():
        gr.Markdown("#### Training Arguments")
        with gr.Row():
            output_dir_input_4 = gr.Textbox(label="Output Directory", value="./output")
            max_steps_input_4 = gr.Slider(label="Max Steps", minimum=100, maximum=10000, step=100, value=500)
            learning_rate_input_4 = gr.Number(label="Learning Rate", value=3e-3, precision=4)
            per_device_train_batch_size_input_4 = gr.Slider(label="Train Batch Size", minimum=1, maximum=64, step=1,
                                                            value=4)
        with gr.Row():
            dataloader_num_workers_input_4 = gr.Slider(label="Data Loader Workers", minimum=1, maximum=32, step=1,
                                                       value=16)
            remove_unused_columns_input_4 = gr.Checkbox(label="Remove Unused Columns", value=False)
            save_strategy_input_4 = gr.Dropdown(label="Save Strategy", choices=["steps", "epoch"], value="steps")
            save_steps_input_4 = gr.Slider(label="Save Steps", minimum=50, maximum=1000, step=50, value=100)

    # Logging & Evaluation Section
    with gr.Group():
        gr.Markdown("#### Logging & Evaluation")
        with gr.Row():
            log_level_input_4 = gr.Dropdown(label="Log Level", choices=["debug", "info", "warn", "error"], value="info")
            logging_strategy_input_4 = gr.Dropdown(label="Logging Strategy", choices=["steps", "epoch"], value="steps")
            logging_steps_input_4 = gr.Slider(label="Logging Steps", minimum=10, maximum=500, step=10, value=10)
        with gr.Row():
            per_device_eval_batch_size_input_4 = gr.Slider(label="Eval Batch Size", minimum=1, maximum=64, step=1,
                                                           value=16)
            eval_strategy_input_4 = gr.Dropdown(label="Eval Strategy", choices=["steps", "epoch"], value="steps")
            eval_steps_input_4 = gr.Slider(label="Eval Steps", minimum=50, maximum=1000, step=50, value=100)
            predict_with_generate_input_4 = gr.Checkbox(label="Predict with Generate", value=True)
            max_new_tokens_input_4 = gr.Slider(label="Max New Tokens", minimum=128, maximum=1024, step=64, value=512)

    # PEFT Configuration Section
    with gr.Group():
        gr.Markdown("#### PEFT Configuration")
        with gr.Row():
            peft_type_input_4 = gr.Dropdown(label="PEFT Type", choices=["PREFIX_TUNING", "ADAPTER", "BITFIT"],
                                            value="PREFIX_TUNING")
            task_type_input_4 = gr.Dropdown(label="Task Type", choices=["CAUSAL_LM", "SEQ2SEQ_LM"], value="CAUSAL_LM")
        with gr.Row():
            num_virtual_tokens_input_4 = gr.Slider(label="Num Virtual Tokens", minimum=128, maximum=1024, step=64,
                                                   value=512)
            num_attention_heads_input_4 = gr.Slider(label="Num Attention Heads", minimum=1, maximum=16, step=1, value=2)
            token_dim_input_4 = gr.Slider(label="Token Dimension", minimum=128, maximum=1024, step=64, value=256)

    # Output Configuration Section
    with gr.Group():
        gr.Markdown("#### Output Configuration")
        config_output_path_input_4 = gr.Textbox(label="Enter ./Step4_output_path/config.yaml")
        block4_button = gr.Button("Start Block4", variant="primary")

    block4_output = gr.Textbox(label="YAML Generation Result", lines=1, max_lines=4, interactive=False)

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
