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

# 创建一个 threading.Event 对象来代替 stop_flag
stop_event_block1 = threading.Event()

def clean_text_block1(soup):
    # 删除常见的无关部分，比如导航栏、页脚、广告、弹窗、侧边栏、脚本和样式
    for element in soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style', 'noscript', 'iframe', 'embed', 'object', 'form', 'input', 'button']):
        element.extract()

    # 进一步过滤特定的 div 或者 span 等标签，这些标签通常包含无关内容
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

    # 过滤掉含有特定 id 或 class 的元素
    for element in soup.find_all(lambda tag: 'cookie' in tag.get('id', '') or 'cookie' in tag.get('class', []) or
                                             'modal' in tag.get('id', '') or 'modal' in tag.get('class', []) or
                                             'popup' in tag.get('id', '') or 'popup' in tag.get('class', []) or
                                             'subscribe' in tag.get('id', '') or 'subscribe' in tag.get('class', [])):
        element.extract()

    # 替换常见的 HTML 实体，如 &nbsp;
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

    # 更严格的域名匹配，确保完整匹配
    return href_domain.endswith(base_domain)

def extract_text_from_html_block1(url, max_pages, output_dir):
    stop_event_block1.clear()  # 每次开始抓取时重置停止标志
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

        if stop_event_block1.is_set():  # 使用 stop_event_block1 检查是否停止
            print("程序终止！请重新运行此BLOCK！")
            return

        if current_url in visited_urls or depth >= max_pages or page_count >= max_pages:
            return
        visited_urls.add(current_url)

        try:
            # 设置 verify=True 以确保 SSL 证书验证
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

        # 使用 BeautifulSoup 清理 HTML 并提取文本
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

# Gradio interface setup
with gr.Blocks(title="UMTCDS") as demo:
    # Block 1
    gr.Markdown("### Block1-从目标URL中获取初始数据 ")
    gr.Markdown("请输入 #URL，代码会自动递归访问URL的第一层子URL，每个URL都将被转换为txt格式文档存储在你的 #输出位置 ")
    with gr.Group():
        with gr.Row():
            url_input_block1 = gr.Textbox(label="URL", placeholder="样例：https://fie.must.edu.mo/index.html?locale=zh_CN",
                                   elem_id="url-input-block1")
            output_dir_input_block1 = gr.Textbox(label="输出位置", placeholder="请键入 txt的输出位置，可使用绝对路径或相对路径",
                                          elem_id="output-dir-input-block1")
            max_pages_input_block1 = gr.Slider(label="URL访问次数", minimum=1, maximum=100, step=1, value=50,
                                        elem_id="max-pages-input-block1")

        with gr.Row():
            submit_button_block1 = gr.Button("运行 BLOCK1 ", variant="primary", elem_id="submit-button-block1")
            stop_button_block1 = gr.Button("终止 BLOCK1", variant="secondary", elem_id="stop-button-block1")

        output_block1 = gr.Textbox(label="实时日志", lines=5, interactive=False, elem_id="output-box-block1")

    submit_button_block1.click(run_block1, inputs=[url_input_block1, max_pages_input_block1, output_dir_input_block1], outputs=output_block1)
    stop_button_block1.click(stop_extraction_block1, outputs=output_block1)

    demo.launch()
