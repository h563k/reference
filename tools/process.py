from tools.EsPdfProcess import PDFProcessor
from tools.QdDatabast import QdrantDatabase
from tools.EsDataset import ESDataset
from tools.recall import recall
from tools.rag import rag_main

def main():
    # 处理文献，写入 es 数据库
    es = PDFProcessor()
    #  处理文献，写入向量数据库
    qd = QdrantDatabase()
    es_dataset = ESDataset()
    es.process_all()
    qd.process_all()
    recall(es_dataset, qd)
    rag_main()
    
