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

    def detect_language(self, file_path):
        """根据文件路径判断语言"""
        return "chinese" if re.search(r'[\u4e00-\u9fa5]', file_path) else "english"

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
        patterns = {
            "english": r'(\b\d+(\.\d+)+\s+[^\n]+\n?)',
            "chinese": r'(\b\d+(\.\d+)+\s+[^\n]+\n?)|([零一二三四五六七八九十百]+、)'
        }

        sections = []
        pattern = re.compile(patterns[language], re.MULTILINE)

        # 获取所有标题位置
        matches = list(pattern.finditer(text))
        print("匹配的matches为",matches)
        if not matches:
            return []

        # 构建标题区块
        last_end = 0
        for match in matches:
            start_pos = match.start()
            if start_pos > last_end:
                sections.append({
                    "type": "content",
                    "text": text[last_end:start_pos],
                    "page_num": page_num
                })
            sections.append({
                "type": "title",
                "text": match.group(),
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
        section_keywords = {
            "english": ["abstract", "introduction", "conclusion"],
            "chinese": ["摘要", "引言", "结论"]
        }

        sections = []
        current_section = {"type": "other", "title": "", "content": []}

        for seg in segments:
            if seg['type'] == 'title':
                # 检查是否是特殊章节
                lower_text = seg['text'].lower()
                language = "chinese" if re.search(
                    r'[\u4e00-\u9fa5]', lower_text) else "english"

                for kw in section_keywords[language]:
                    if kw in lower_text:
                        # 保存当前章节并开始新章节
                        if current_section["content"]:
                            sections.append(current_section)
                        current_section = {
                            "type": kw,
                            "title": seg['text'],
                            "content": [],
                            "start_page": seg['page_num']
                        }
                        break
            else:
                current_section["content"].append(seg['text'])

        # 添加最后一个章节
        if current_section["content"]:
            sections.append(current_section)

        return sections

    def process_file(self, file_path):
        """处理单个PDF文件"""
        # 检测语言
        language = self.detect_language(file_path)
        file_name = os.path.basename(file_path)

        # 提取文本
        pages = self.extract_text_by_page(file_path)

        # 分段处理
        all_segments = []
        for page in pages:
            if segments := self.split_by_titles(page['text'], page['page_num'], language):
                all_segments.extend(segments)
            else:  # 降级为自然段切割
                paragraphs = [p.strip()
                              for p in page['text'].split('\n\n') if p.strip()]
                all_segments.extend({
                    "type": "paragraph",
                    "text": p,
                    "page_num": page['page_num']
                } for p in paragraphs)

        # 提取特殊章节
        special_sections = self.extract_special_sections(all_segments)

        # 构建ES文档
        documents = []
        for section in special_sections:
            segment_text = " ".join(section["content"])
            if segment_text:
                documents.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "language": language,
                    "section_type": section["type"],
                    "section_title": section["title"],
                    "segment_text": self.clean_text(segment_text),
                    "page_num": section["start_page"]
                })

        # 写入ES
        if documents:
            for doc in documents:
                self.es.index(index=self.index_name, document=doc)
            print(f"成功写入 {len(documents)} 个段落: {file_name}")
        return len(documents)

    def process_directory(self, directory_path):
        """批量处理目录中的PDF文件"""
        print(f"开始处理目录: {directory_path}")
        count = 0

        # 获取PDF文件列表
        pdf_files = [f for f in os.listdir(
            directory_path) if f.lower().endswith('.pdf')]

        # 进度显示
        pbar = tqdm(pdf_files, desc="处理PDF文件")
        for filename in pbar:
            file_path = os.path.join(directory_path, filename)
            count += self.process_file(file_path)
            pbar.set_postfix(file=filename[:20])

        print(f"处理完成! 共处理 {len(pdf_files)} 个文件, 写入 {count} 个段落")


# 使用示例
if __name__ == "__main__":
    processor = PDFProcessor(
        index_name="pdf_segments"      # 自定义索引名
    )

    # 处理单个文件
    processor.process_file("/opt/project/reference/origin_data/李欣懿-参考文献集合-推荐/参考文献/[1]地球内部的氧化还原地球动力.pdf")

    # 处理整个目录
    # processor.process_directory("/文献路径/中文文献/")
