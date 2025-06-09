from tools.EsPdfProcess import PDFProcessor
from tools.QdDatabast import QdrantDatabase


def main():
    es = PDFProcessor()
    es.process_all()
    qd = QdrantDatabase()
    qd.process_all()
