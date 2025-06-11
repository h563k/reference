from tools.EsPdfProcess import PDFProcessor
from tools.QdDatabast import QdrantDatabase


def main():
    # 处理文献，写入 es 数据库
    es = PDFProcessor()
    #  处理文献，写入向量数据库
    qd = QdrantDatabase()
    es.process_all()
    qd.process_all()
    
