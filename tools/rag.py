import os
import re
import openai
import json
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
load_dotenv()


def key_word_query(query: str, data_query: str, reference_name: str) -> str:
    system_prompt = f"""你是一个学术诚信审核专家，需要严格判断参考文献的引用质量。请按照以下规则分类：
【0 虚假引用】文献原话的关键信息在片段中完全不存在，或与片段内容相矛盾
【1 不当引用】文献原话与片段存在部分关联，但存在断章取义、过度解读或扭曲原意
【2 正确引用】文献原话的核心主张、数据、结论与片段内容完全一致

# 分析步骤
1. 对比数据：逐句核对文献原话（Query）与参考文献片段（Data_Query）的内容关联性
2. 识别关键信息：特别关注数据、结论、专有名词和因果关系等关键元素
3. 判断失真类型：若存在分歧，分析是虚构内容（选0）还是曲解原意（选1）
4. 最终验证：只有当核心含义完全匹配时才判定为正确引用
5. 忽略其中的无意义符合，如

# 输入格式
Query: "待验证的文献原话"
Reference_name: "引用的其中一篇参考文献"
Data_Query: "从参考文献中搜索到的原始片段"

# 输出格式
严格按此JSON格式输出，不要解释：
"verdict": 分类编号, "evidence": "核心矛盾点的简要说明"
"""
    prompt = f"""Query: "{query}"
Reference_name: "{reference_name}"
Data_Query: "{data_query}"
"""
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
        temperature=0,
    )
    return response.choices[0].message.content


def reranker(query_str: str, query_list: str):
    # 检查GPU可用性并选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 初始化模型和tokenizer
    model_name = "BAAI/bge-reranker-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()  # 设置为评估模式
    # 我们将对每个文档进行评分
    pairs = [[query_str, doc] for doc in query_list]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True,
                           return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        # 将分数转换为列表
        scores = scores.tolist()

        # 将文档和分数组合在一起
        doc_score_pairs = list(zip(query_list, scores))

        # 根据分数降序排序
        doc_score_pairs_sorted = sorted(
            doc_score_pairs, key=lambda x: x[1], reverse=True)

        # 如果我们只需要排序后的文档
        sorted_documents = [doc for doc, score in doc_score_pairs_sorted]

    return sorted_documents


def rag_main():
    result_file = 'data/final_result.jsonl'  # 使用.jsonl扩展名表示JSON Lines格式
    # 先清空文件（可选，根据是否需要保留历史数据决定）
    open(result_file, 'w').close()

    with open('data/data_query.json', 'r') as f:
        data = json.load(f)

    for reference in data:
        for ref in reference.values():
            reference = ref  # 重命名避免变量覆盖

        # 文献原话， 详细引用
        for query_sentence, sentencr_detail in reference.items():
            for reference_name, query_list in sentencr_detail.items():
                sorted_documents = reranker(query_sentence, query_list)
                max_length = 3000
                lens = 0
                document = ""
                for doc in sorted_documents:
                    lens += len(doc)
                    if lens < max_length:
                        document += doc
                    else:
                        document += doc[:max_length - lens]
                response = key_word_query(
                    query_sentence, document, reference_name)
                verdict = re.findall(r'"verdict": (\d+)', response)
                evidence = re.findall(r'"evidence": "(.+?)"', response)
                response = {
                    "verdict": verdict[0] if verdict else None,
                    "evidence": evidence[0] if evidence else None
                }
                current_result = {
                    "query_sentence": query_sentence,
                    "reference_name": reference_name,
                    "response": response  # 包含verdict和evidence
                }    
                # 实时写入单条结果到文件（JSON Lines格式）
                with open(result_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(current_result,
                            ensure_ascii=False) + '\n')


if __name__ == '__main__':
    rag_main()
