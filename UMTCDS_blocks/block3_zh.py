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
import os
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor


# 去除引号的函数（与block3相关）
def block3_remove_quotes(text):
    return re.sub(r'[“”""]', '', text)


# 验证输入路径和文件名的函数（与block3相关）
def block3_validate_inputs(input_folder, base_folder, filenames):
    if not os.path.exists(input_folder):
        raise ValueError(f"输入目录： {input_folder} 不存在！")
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    for filename in filenames:
        if not filename.endswith('.jsonl'):
            raise ValueError(f"文件名非法： {filename}. 必须以 '.jsonl'结尾")


# 多线程并行处理多个文件中的问答对（与block3相关）
def block3_process_file(file_path):
    all_contents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            all_contents.extend(content.split("\n\n"))
    except Exception:
        pass  # 如果有错误，这里简单跳过
    return all_contents


# 使用多线程提取问答对（与block3相关）
def block3_extract_qa_pairs_parallel(input_folder):
    """使用多线程并行处理多个文件"""
    with ThreadPoolExecutor() as executor:
        file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                      os.path.isfile(os.path.join(input_folder, f))]
        results = list(executor.map(block3_process_file, file_paths))
    return [qa_pair for result in results for qa_pair in result]  # 合并结果


# 转换为JSONL并拆分为训练集、验证集和测试集（与block3相关）
def block3_convert_to_jsonl(input_folder, base_folder, train_filename, val_filename, test_filename):
    # 验证输入的路径和文件名
    block3_validate_inputs(input_folder, base_folder, [train_filename, val_filename, test_filename])

    # 输出文件路径配置
    output_files = {
        "train": os.path.join(base_folder, train_filename),
        "val": os.path.join(base_folder, val_filename),
        "test": os.path.join(base_folder, test_filename)
    }

    all_contents = block3_extract_qa_pairs_parallel(input_folder)

    # 随机打乱数据
    random.shuffle(all_contents)

    # 计算拆分点
    total_count = len(all_contents)
    train_split = int(total_count * 0.8)
    val_split = train_split + int(total_count * 0.1)

    # 分别写入训练集、验证集和测试集
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

            # 写入对应文件
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


# Gradio接口部分（与block3相关）
import gradio as gr

with gr.Blocks(title="UMTCDS") as demo:
    gr.Markdown("### Block-3  生成JSONL数据集")
    gr.Markdown("该部分将批量转换txt文件为三个JSONL数据集文件。")
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


    def run_block3(input_folder, base_folder, train_filename, val_filename, test_filename):
        log = block3_convert_to_jsonl(input_folder, base_folder, train_filename, val_filename, test_filename)
        yield log


    block3_button.click(run_block3,
                        inputs=[block3_input_path, block3_base_folder, block3_train_filename, block3_val_filename,
                                block3_test_filename], outputs=block3_output)

    demo.launch(server_name="0.0.0.0", server_port=6006)