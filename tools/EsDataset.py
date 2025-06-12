import os
import logging
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 设置Elasticsearch日志级别为DEBUG（可选）
# logging.basicConfig(level=logging.DEBUG)


class ESDataset:
    def __init__(self, index_name='pdf_segments'):
        self.es = self.es_login()
        self.index_name = index_name
        if not self.es.indices.exists(index=self.index_name):
            self._create_index()

    def es_login(self):
        # 打印环境变量值
        db_url = str(os.getenv("DATABASE_URL"))
        user = str(os.getenv("USER"))
        password = str(os.getenv("PASSWORD"))
        print(f"Connecting to Elasticsearch at: {db_url} with user: {user}")
        es = Elasticsearch(
            hosts=[db_url],
            basic_auth=(user, password),
            verify_certs=True,
            ca_certs="/opt/project/reference/certs/http_ca.crt",
            ssl_assert_hostname=False,
        )
        # 检查连接
        print(f"Elasticsearch client info: {es.info()}")
        return es

    @property
    def client(self):
        return self.es

    def _create_index(self):
        """Create the Elasticsearch index with appropriate mapping"""
        mapping = {
            "mappings": {
                "properties": {
                    "file_path": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "section_type": {"type": "keyword"},
                    "section_title": {"type": "text"},
                    "segment_text": {"type": "text"},
                    "page_num": {"type": "integer"},
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"Created index: {self.index_name}")


# 在ESDataset类下方添加以下实现
class ESRetriever(BaseRetriever):
    def __init__(self, es_dataset: ESDataset):
        super().__init__()
        self.es = es_dataset.client
        self.index_name = es_dataset.index_name

    # 修改参数
    def _retrieve(self, query_str: str, file_name: str) -> List[NodeWithScore]:
        query = {
            "bool": {
                "must": {
                    "match": {"segment_text": query_str}  # 直接使用查询字符串
                },
                "filter": {
                    "term": {"file_name": file_name}  # 直接使用文件名参数
                }
            }
        }

        res = self.es.search(
            index=self.index_name,
            body={"query": query},
            size=3
        )

        nodes = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            node = TextNode(
                text=source["segment_text"],
                metadata={
                    "file_name": source["file_name"],
                }
            )
            nodes.append(NodeWithScore(node=node, score=hit["_score"]))
        return nodes

    def retrieve(self, query_str: str, file_name: str) -> List[NodeWithScore]:
        # 直接传递参数到内部方法
        return self._retrieve(query_str, file_name)


def query_es(query_str: str, file_name: str, es_dataset: ESDataset) -> List[NodeWithScore]:
    # 创建检索器
    retriever = ESRetriever(es_dataset)
    # 执行带过滤的检索
    results = retriever.retrieve(
        query_str,
        file_name
    )
    return results
