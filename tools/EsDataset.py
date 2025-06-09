import os
import logging
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 设置Elasticsearch日志级别为DEBUG（可选）
logging.basicConfig(level=logging.DEBUG)


class ESDataset:
    def __init__(self, index_name='pdf_segments'):
        self.es = self.es_login()
        self.index_name = index_name
        if not self.es.indices.exists(index=self.index_name):
            self._create_index()

    def es_login(self):
        # 打印环境变量值
        db_url = os.getenv("DATABASE_URL")
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
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


class ElasticsearchKeywordRetriever(BaseRetriever):
    def __init__(self, index_name: str, content_field: str):
        self.es = ESDataset(index_name).client
        self.index_name = index_name
        self.content_field = content_field

    def _retrieve(self, query_bundle: QueryBundle, **kwargs) -> List[NodeWithScore]:
        # 获取查询字符串
        query_str = query_bundle.query_str.strip()
        if not query_str:
            return []

        # 构建关键词搜索查询
        es_query = {
            "query": {
                "match": {
                    self.content_field: {
                        "query": query_str,
                        "operator": "and"  # 要求所有关键词都匹配
                    }
                }
            },
            "size": 10  # 限制返回结果数量
        }

        try:
            # 执行Elasticsearch查询
            response = self.es.search(
                index=self.index_name,
                body=es_query
            )
        except Exception as e:
            print(f"Elasticsearch查询失败: {e}")
            return []

        # 处理搜索结果
        results = []
        for hit in response['hits']['hits']:
            doc = hit['_source']
            # 创建文本节点
            text_node = TextNode(
                text=doc.get(self.content_field, ""),
                id_=hit['_id'],
                metadata={
                    "document_id": hit['_id'],
                    "score": hit['_score']
                }
            )
            # 包装为带分数的节点
            results.append(NodeWithScore(
                node=text_node,
                score=hit['_score']
            ))

        return results


if __name__ == "__main__":
    retriever = ElasticsearchKeywordRetriever(
        index_name="pdf_segments", content_field="content")
    query_bundle = QueryBundle(query_str="如何使用llama_index")
    results = retriever.retrieve(query_bundle)
    for result in results:
        print(result.node.text)
