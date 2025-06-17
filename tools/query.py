import os
import re
import json
import openai
from tools.EsDataset import query_es, ESDataset
from dotenv import load_dotenv
from tools.WordProcess import process_all_word
load_dotenv()
es_dataset = ESDataset()
articles = process_all_word()


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


def query_es_by_keywords():
    """根据关键词查询ES数据库
    results: [
        {
            'article_name': {
                'sentence_name': {
                    'reference_name': [
                        {
                            'text': '句子内容',
                            'score': 0.9
                        }
                    ]
                }
                },
        }
    ]
    """
    results = []
    # 文章合集
    for article in articles:
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
                    # 每句话的关键词
                    sentence_result = []
                    for key_word in key_words:
                        # 查询结果
                        results = query_es(key_word, reference, es_dataset)
                        results = [{'text': x.text, 'score': x.score}
                                   for x in results]
                        sentence_result.extend(results)
                    reference_query[reference] = sentence_result
                article_temp[text] = reference_query
            article_name_set[article_name] = article_temp
        results.append(article_name_set)
    return results


def query_main():
    with open("data_query.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        results = query_es_by_keywords()
        with open("data_query.json", "w", encoding="utf-8") as f:
            json.dump(results, f)

    print(results)
