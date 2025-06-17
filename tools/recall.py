import os
import json
import re
import openai
from tools.EsDataset import query_es, ESDataset
from dotenv import load_dotenv
from tools.WordProcess import process_all_word
from tools.QdDatabast import QdrantDatabase
load_dotenv()
es_dataset = ESDataset()
db = QdrantDatabase()
articles = process_all_word()


def qd_word_query(query_str: str, reference: str, db: QdrantDatabase):
    retriever_filtered = db.get_retriever(
        source_files=[reference])
    results_filtered = retriever_filtered.retrieve(query_str)
    return [x.node.text for x in results_filtered]


def key_word_query(prompt: str, language: str) -> str:
    system_prompt = f"""你是一位地质学文献检索专家。请从以下地质学句子中提取用于数据库查询的核心关键词和短语，目的是查找该句所引用的参考文献。要求：
1.  必须包含句中提到的具体地质构造名称、重要地理区域名称和核心地质概念短语。
2.  输出的语言为{language}
3.  排除连接词、普通动词、修饰性的非专业形容词（除非构成专有名词一部分）。
4.  请用逗号分隔输出结果。"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.base_url = os.getenv("OPENAI_BASE_URL")
    model = str(os.getenv("OPENAI_MODELS"))
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content


def detect_language(text):
    """根据文本内容判断语言"""
    return "中文" if re.search(r'[\u4e00-\u9fa5]', text) else "英文"


def recall(es_dataset, db):
    with open("data/data_query.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    start = len(results)
    # 文章合集
    for i, article in enumerate(articles):
        if i < start:
            continue
        # 单篇文章参数
        article_name_set = {}
        for article_name, references_list in article.items():
            # 文章引用参考文献的话
            article_temp = {}
            for text, reference_list in references_list.items():
                # 每句话引用参考文献
                reference_query = {}
                for reference in reference_list:
                    language = detect_language(reference)
                    key_words = key_word_query(text, language).split(",")
                    key_words = " ".join(key_words)
                    # 每句话的关键词
                    sentence_result = []
                    # 查询结果
                    sentence_result_query = query_es(
                        key_words, reference, es_dataset)
                    sentence_result_query = [
                        x.text for x in sentence_result_query]
                    qd_word = qd_word_query(text, reference, db)
                    sentence_result.extend(sentence_result_query)
                    sentence_result.extend(qd_word)
                    reference_query[reference] = sentence_result
                article_temp[text] = reference_query
            article_name_set[article_name] = article_temp
        results.append(article_name_set)
        with open("data/data_query.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    return results
