import os
import re
import glob
from docx import Document


def parse_citation(citation_str):
    """
    解析引用字符串，提取所有文献序号（包括扩展范围）

    参数:
        citation_str: 引用字符串（如'[1-3]'）

    返回:
        list: 所有文献序号的整数列表
    """
    # 去掉方括号
    inner = citation_str.strip('[]')

    # 拆分逗号分隔的部分
    parts = re.split(r',\s*', inner)

    # 存储结果的集合（自动去重）
    results = set()

    for part in parts:
        # 处理连字符范围（如1-3）
        if '-' in part:
            start, end = re.split(r'\s*-\s*', part)
            if start.isdigit() and end.isdigit():
                start, end = int(start), int(end)
                # 确保范围有效
                if start <= end:
                    results.update(range(start, end + 1))
        # 处理单个数字（如1,2]
        elif part.isdigit():
            results.add(int(part))

    return sorted(results)


def extract_references_and_citations(doc_path):
    doc = Document(doc_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
    split_text = re.findall('引\x20+言', full_text)[0]
    if split_text:
        full_text = full_text.split(split_text)[1:]
        full_text = 'split_text'.join(full_text)
    full_text = full_text.split('参考文献')
    full_text = full_text[:-1]
    full_text = '参考文献'.join(full_text)
    full_text = full_text.split('。')
    reference_dict = {}
    for i, text in enumerate(full_text):
        text = text+'。'
        text_num = re.findall(r'\[.*?\]', text)
        if not text_num:
            continue
        text_num_list = []
        for x in text_num:
            text_num_list.extend(parse_citation(x))
        text_num_list = list(set(text_num_list))
        text_num_list.sort()
        reference_dict[text] = text_num_list
    return reference_dict


def reference_process(doc_path):
    reference_dict = extract_references_and_citations(doc_path)
    reference_path = os.path.join(
        os.path.dirname(os.path.dirname(doc_path)), '参考文献')
    path_list = os.listdir(reference_path)
    reference_dict_text = {}
    for path in path_list:
        match_path = re.match(r'\[(\d+)\]', path)
        if not match_path:
            print(f'以下路径解析失败{path}')
            continue
        reference_dict_text[match_path.group(1)] = path
    reference_dict_new = {}
    for text, num_list in reference_dict.items():
        reference_dict_new[text] = []
        for x in num_list:
            if not reference_dict_text.get(str(x)):
                continue
            name = reference_dict_text[str(x)]
            reference_dict_new[text].append(name)
    return reference_dict_new


def process_folder(folder_path):
    # 处理所有doc和docx文件
    for file_path in glob.glob(os.path.join(folder_path, '*.doc*')):
        if file_path.endswith('.doc') or file_path.endswith('.docx'):
            reference_dict = reference_process(file_path)
            return reference_dict


def process_all_word():
    reference = []
    path_list = os.listdir('origin_data')
    path_list = [os.path.join('origin_data', x, '作者文稿') for x in path_list]
    for path in path_list:
        file_name = os.path.dirname(path)
        file_name = os.path.basename(file_name)
        print(f"正在处理文件夹: {file_name}")
        reference_dict = process_folder(path)
        reference.append({file_name: reference_dict})
    return reference


if __name__ == '__main__':
    # folder_path = "/opt/project/reference/origin_data/魏超凡-参考文献集合-推荐/作者文稿"
    # process_folder(folder_path)
    process_all_word()
