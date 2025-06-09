import os
import re
import fitz  # PyMuPDF
from tools.EsDataset import ESDataset
from collections import defaultdict
from tqdm import tqdm


class PDFProcessor:
    def __init__(self, index_name='pdf_segments'):
        self.es = ESDataset().client
        self.index_name = index_name
        # 定义中英文特殊章节关键词映射
        self.section_keywords = {
            "english": {"abstract": ["abstract"],
                        "introduction": ["introduction"],
                        "conclusion": ["conclusion"]},
            "chinese": {"摘要": ["摘要"],
                        "引言": ["引言"],
                        "结论": ["结论"]}
        }

    def detect_language(self, text):
        """根据文本内容判断语言"""
        return "chinese" if re.search(r'[\u4e00-\u9fa5]', text) else "english"

    def extract_text_by_page(self, file_path):
        """提取PDF每页文本（保留页码信息）"""
        doc = fitz.open(file_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pages.append({
                "page_num": page_num + 1,
                "text": page.get_text("text")  # 提取纯文本
            })
        return pages

    def clean_text(self, text):
        """清理文本中的多余空格/换行"""
        return re.sub(r'\s+', ' ', text).strip()

    def split_by_titles(self, text, page_num, language):
        """按层级标题切割内容"""
        # 改进后的正则表达式，能识别更多类型的标题格式
        chinese_pattern = r'([零一二三四五六七八九十百]+、)|\s*(\d+(\.\d+)+\s+[^\n]+)'
        english_pattern = r'(\b\d+(\.\d+)+\s+[^\n]+)|(^[A-Z][^\n]+$)'

        pattern = chinese_pattern if language == "chinese" else english_pattern
        compiled_pattern = re.compile(pattern, re.MULTILINE)

        matches = list(compiled_pattern.finditer(text))
        if not matches:
            return []

        sections = []
        last_end = 0

        for match in matches:
            start_pos = match.start()
            if start_pos > last_end:
                sections.append({
                    "type": "content",
                    "text": text[last_end:start_pos],
                    "page_num": page_num
                })

            # 提取实际的匹配文本（可能在不同的分组中）
            title_text = next((group for group in match.groups() if group), "")
            sections.append({
                "type": "title",
                "text": title_text.strip(),
                "page_num": page_num
            })
            last_end = match.end()

        # 添加剩余内容
        if last_end < len(text):
            sections.append({
                "type": "content",
                "text": text[last_end:],
                "page_num": page_num
            })

        return sections

    def extract_special_sections(self, segments):
        """识别并标记特殊章节（摘要/介绍等）"""
        sections = []
        current_section = {"type": "other",
                           "title": "", "content": [], "start_page": 1}

        for seg in segments:
            if seg['type'] == 'title':
                lower_text = seg['text'].lower()
                language = "chinese" if re.search(
                    r'[\u4e00-\u9fa5]', lower_text) else "english"

                found_special = False
                for section_type, keywords in self.section_keywords[language].items():
                    if any(kw in lower_text for kw in keywords):
                        # 保存当前章节并开始新章节
                        if current_section["content"]:
                            sections.append(current_section)

                        current_section = {
                            "type": section_type,
                            "title": seg['text'],
                            "content": [],
                            "start_page": seg['page_num']
                        }
                        found_special = True
                        break

                # 如果是特殊章节，标题不加入内容
                if found_special:
                    continue

            # 将内容添加到当前章节
            if seg['text'].strip():
                # 对于内容类型，如果当前章节还没有设置起始页，则设置为当前页
                if seg['type'] == 'content' and 'start_page' not in current_section:
                    current_section['start_page'] = seg['page_num']

                current_section["content"].append(seg['text'])

        # 添加最后一个章节
        if current_section["content"]:
            sections.append(current_section)

        return sections

    def process_file(self, file_path):
        """处理单个PDF文件"""
        file_name = os.path.basename(file_path)

        # 提取文本
        pages = self.extract_text_by_page(file_path)

        # 如果整个文件都没内容，直接返回
        if not pages or not any(p['text'].strip() for p in pages):
            print(f"跳过无内容的文件: {file_name}")
            return 0

        # 从第一页内容判断语言
        language = self.detect_language(pages[0]['text'])
        print(f"文件 {file_name} 语言: {language}")

        # 分段处理
        all_segments = []
        for page in pages:
            # 跳过空页
            if not page['text'].strip():
                continue

            segments = self.split_by_titles(
                page['text'], page['page_num'], language)
            if segments:
                all_segments.extend(segments)
            else:
                # 降级处理：按段落分割
                paragraphs = re.split(r'\n\s*\n', page['text'])
                all_segments.extend({
                    "type": "paragraph",
                    "text": p.strip(),
                    "page_num": page['page_num']
                } for p in paragraphs if p.strip())

        print(f"文件 {file_name} 分段数: {all_segments}")
        # 提取特殊章节
        special_sections = self.extract_special_sections(all_segments)
        print(f"文件 {file_name} 特殊章节数: {len(special_sections)}")
        # 构建ES文档
        documents = all_segments
        for i, section in enumerate(special_sections):
            # 确保至少有起始页
            start_page = section.get(
                'start_page', all_segments[-1]['page_num']+i+1)
            if section['type'] != 'other':  # 排除其他类型
                segment_text = " ".join(section["content"]).strip()
                if segment_text:
                    documents.append({
                        "file_path": file_path,
                        "file_name": file_name,
                        "language": language,
                        "section_type": section["type"],
                        "section_title": section["title"],
                        "segment_text": self.clean_text(segment_text),
                        "page_num": start_page
                    })

        # 写入ES
        if documents:
            for doc in documents:
                self.es.index(index=self.index_name, document=doc)
            print(f"成功写入 {len(documents)} 个段落: {file_name}")
        else:
            print(f"未发现可索引内容: {file_name}")

        return len(documents)

    def process_directory(self, directory_path):
        """批量处理目录中的PDF文件"""
        print(f"开始处理目录: {directory_path}")
        count = 0

        # 获取PDF文件列表
        pdf_files = [f for f in os.listdir(directory_path)
                     if f.lower().endswith('.pdf')]

        # 进度显示
        pbar = tqdm(pdf_files, desc="处理PDF文件")
        for filename in pbar:
            file_path = os.path.join(directory_path, filename)
            try:
                count += self.process_file(file_path)
                pbar.set_postfix(file=filename[:20], segments=count)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

        print(f"处理完成! 共处理 {len(pdf_files)} 个文件, 写入 {count} 个段落")


# 使用示例
if __name__ == "__main__":
    processor = PDFProcessor(index_name="pdf_segments")

    # 处理单个文件
    processor.process_file(
        "/opt/project/reference/origin_data/李欣懿-参考文献集合-推荐/参考文献/[1]地球内部的氧化还原地球动力.pdf")

    # 处理整个目录
    # processor.process_directory("/文献路径/中文文献/")
