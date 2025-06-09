from tools.pdf_process import PDFProcessor
from tools.EsDataset import ElasticsearchKeywordRetriever
processor = PDFProcessor(
    index_name="pdf_segments"      # 自定义索引名
)

# 处理单个文件
processor.process_file("/opt/project/reference/origin_data/李欣懿-参考文献集合-推荐/参考文献/[1]地球内部的氧化还原地球动力.pdf")

# from llama_index.core import QueryBundle

# retriever = ElasticsearchKeywordRetriever(
#     index_name="pdf_segments",
#     content_field="content"
# )
# query_bundle = QueryBundle(query_str="地球")
# results = retriever.retrieve(query_bundle)
# for result in results:
#     print(result.node.text)
