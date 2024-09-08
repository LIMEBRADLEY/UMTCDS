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
        log = f"API key无效或无法初始化ZhipuAI客户端: {e}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    # 确保保存路径存在
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

    # 检查输入文件夹是否存在并且包含txt文件
    if not os.path.exists(block2a_folder_path):
        log = f"输入文件夹路径不存在: {block2a_folder_path}\n"
        logs.append(log)
        yield ''.join(logs)
        return

    block2a_txt_files_found = False

    # 遍历文件夹下的所有子文件夹和txt文件
    for root, dirs, files in os.walk(block2a_folder_path):
        if stop_event_block2a.is_set():
            log = "进程已被中断。\n"
            logs.append(log)
            yield ''.join(logs)
            return

        # 对文件进行排序，以确保按顺序处理
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
                             "content": "请你对以下文本进行清理，去除所有无关内容。然后，将清理后的内容转换为问答对，每个问答对应该准确且与原始文本内容相关。"},
                            {"role": "assistant", "content": "当然，请将您的文本发送给我，我将进行清理并生成问答对，按照（Q： A：）。"},
                            {"role": "user", "content": f"这是我的数据，请帮我整理并只返回给我问答对：{block2a_file_content}"}
                        ],
                    )

                    if not block2a_response.choices or not block2a_response.choices[0].message:
                        log = f"文件 {file} 的处理失败：未收到有效的API响应\n"
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

# 用于处理线程的启动和停止
def run_block2a(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
    logs = ""  # 重置日志内容
    for result in block2a_process_files(block2a_api_key, block2a_folder_path, block2a_output_folder_path):
        logs = result  # 只保留本次结果的日志
        yield logs

def stop_block2a():
    stop_event_block2a.set()  # 设置事件为 True，表示停止
    return "BLOCK2A任务正在停止，请稍候..."

# Gradio interface setup
with gr.Blocks(title="UMTCDS") as demo:
    # Block 2A
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

        block2a_run_button.click(fn=run_block2a, inputs=[block2a_api_key_input, block2a_folder_path_input, block2a_output_folder_path_input], outputs=block2a_log_output)
        block2a_stop_button.click(fn=stop_block2a, outputs=block2a_log_output)

    demo.launch()
