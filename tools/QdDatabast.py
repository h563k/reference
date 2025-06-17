from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import TextNode, BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List, Optional
from tools.config import get_path_list
import json
import os
import re

# 配置类（根据实际情况修改）


class RAGConfig:
    QDRANT_URL = str(os.getenv("QDRANT_URL"))
    COLLECTION_NAME = "pdf_segments"
    EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"

# 增强型Qdrant数据库操作类


class QdrantDatabase:
    def __init__(self):
        self.vector_store = self.initialize_system()
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=RAGConfig.EMBED_MODEL,
            device="cuda",
        )
        # 修改这两个参数
        Settings.chunk_size = 1024  # 窗口大小（单位：token）
        Settings.chunk_overlap = 200  # 窗口重叠量

    def initialize_system(self):
        """初始化向量数据库连接"""
        qdrant_client = QdrantClient(
            url=RAGConfig.QDRANT_URL, prefer_grpc=False)

        # 创建集合（如果不存在）
        try:
            qdrant_client.get_collection(RAGConfig.COLLECTION_NAME)
        except:
            qdrant_client.create_collection(
                collection_name=RAGConfig.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  # multilingual-mpnet-base-v2的维度
                    distance=Distance.COSINE
                )
            )

        return QdrantVectorStore(client=qdrant_client, collection_name=RAGConfig.COLLECTION_NAME)

    def extract_metadata(self, file_path: str) -> dict:
        """从文件名提取语言和其他元数据"""
        filename = os.path.basename(file_path)
        lang = "en"  # 默认英文
        if re.search(r'[\u4e00-\u9fa5]', filename):
            lang = "zh"
        return {"source_file": filename, "language": lang}

    def load_and_chunk_pdfs(self, folder_path) -> List[BaseNode]:
        """加载PDF并分块处理"""
        reader = PyMuPDFReader()
        nodes = []

        for filename in os.listdir(folder_path):
            if not filename.endswith('.pdf'):
                continue
            file_path = os.path.join(folder_path, filename)

            # 提取元数据
            metadata = self.extract_metadata(file_path)

            # 读取PDF文本
            documents = reader.load_data(file_path)

            # 分块处理
            for doc in documents:
                doc.metadata.update(metadata)
                # 创建文本节点（LlamaIndex会自动分块）
                node = TextNode(text=doc.text, metadata=doc.metadata)
                nodes.append(node)
        for i in range(len(nodes)-1, 0, -1):
            text = nodes[i].text.lower()
            if '参考文献' in text:
                lang = "参考文献"
            elif 'reference' in text:
                lang = "reference"
            else:
                continue
            print(f'找到参考文献在第{i}个节点，开始处理...')
            nodes = nodes[:i+1]
            nodes[i] = TextNode(text=text.split(lang)[0], metadata=doc.metadata)
            break

        return nodes

    def ingest_documents(self, file_path):
        """将文献数据写入向量数据库"""
        nodes = self.load_and_chunk_pdfs(file_path)
        # 通过Index添加节点以触发嵌入生成
        index = VectorStoreIndex.from_documents(
            [],
            storage_context=StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        )
        index.insert_nodes(nodes)

    def get_retriever(self, source_files: Optional[List[str]] = None):
        """创建带文献过滤的检索器"""

        # 构建元数据过滤器
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="source_file", value=source_file)
            for source_file in source_files
        ]) if source_files else None

        # 创建带过滤功能的检索器
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
        return index.as_retriever(similarity_top_k=10, filters=filters)

    def process_all(self):
        """处理所有文件"""
        with open("data/file_list.json", 'r', encoding='utf-8') as f:
            file_list = json.load(f)
        path_list = get_path_list()
        for path in path_list:
            if path in file_list['qd']:
                continue
            self.ingest_documents(path)
            file_list['qd'].append(path)
            with open("data/file_list.json", 'w', encoding='utf-8') as f:
                json.dump(file_list, f)


# 使用示例
if __name__ == "__main__":
    # 初始化数据库
    db = QdrantDatabase()

    # STEP 1: 上传文献到向量数据库 (只需运行一次)
    db.ingest_documents(
        '/opt/project/reference/origin_data/李欣懿-参考文献集合-推荐/参考文献')
    print("文献已成功导入向量数据库")

    # STEP 2: 查询时限定文献范围
    query_str = "地球氧化"

    # 选项1: 查询所有文献
    retriever_all = db.get_retriever()

    # 选项2: 只查询特定文献 (e.g., 两个中文文献)
    retriever_filtered = db.get_retriever(
        source_files=["[1]地球内部的氧化还原地球动力.pdf",
                      "[3]Redox Processes Before, During, and After Earth's Accretion Affecting the Deep Carbon Cycle.pdf"]
    )

    # 执行检索
    results_all = retriever_all.retrieve(query_str)
    results_filtered = retriever_filtered.retrieve(query_str)

    print(f"全局检索结果数: {len(results_all)}")
    print(f"限定范围结果数: {len(results_filtered)}")
