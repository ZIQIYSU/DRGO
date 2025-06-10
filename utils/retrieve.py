from nltk.tokenize import sent_tokenize
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def get_yes_or_no(result):
    if 'yes' in str.lower(result)[:5]:return 'Yes'
    if 'no' in str.lower(result)[:5]:return 'No'
    return 'N/A'


def check_score(LLM, context, sentences):
    score_mapping = {'Yes':1.0, 'No':0.0}
    template = """
        Context: {a}
        Sentence: {b}
        Does the context logically and semantically support and implicate the sentence? Answer Yes, No or Uncertainty (Don't give explanations).
        - **Note**: Consider the context and the specific intent of the sentence. Make sure the context really contains or supports the whole sentence.
    """
    scores, results = list(), list()
    for sentence in sentences:
        content = template.format(a=context.strip().replace('/n', ''), b=sentence.strip().replace('/n', ''))
        # print(content)
        result = LLM(content)[0]
        results.append(result)

    results = [get_yes_or_no(r) for r in results]
    scores = [score_mapping.get(result, 0.5) for result in results]

    # for sent, score in zip(sentences, scores):
    #     print(sent.strip(), score)
        #result_string += sent + ' ({a})'.format(a=score)

    return scores


def retrieve_top5(benchmark,em_model, em_tokenizer, query, sentence_embeddings, sentences):
    encoded_query = em_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_embedding = em_model(**encoded_query)[0][:, 0]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    cosine_scores = cosine_similarity(query_embedding.numpy(), sentence_embeddings)

    top_5_indices = np.argsort(cosine_scores[0])[-5:][::-1]
    top_5_results = [sentences[idx] for idx in top_5_indices]
    return top_5_results


def retrieve_queries(benchmark, em_model, em_tokenizer, answer, sentence_embeddings, sentences):
    queries = re.findall(r'\[STA\](.*?)\[END\]', answer)
    all_retrieves = [result for query in queries for result in retrieve_top5(benchmark, em_model, em_tokenizer, query, sentence_embeddings, sentences)]
    if 'arc' in benchmark or 'openbook' in benchmark:
        unique_retrieves =list(dict.fromkeys(all_retrieves))
    elif 'commonsense' in benchmark or 'piqa' in benchmark or 'fever' in benchmark:
        unique_retrieves = list({tuple(sorted(d.items())): d for d in all_retrieves}.values())
    return unique_retrieves


def sample_answer(LLM, query, num):
    answers = list()
    for _ in range(num):
        answer = LLM(query)
        answers.append(answer[0])
    return answers


def format_output(sentences, all_scores, avg_confidence):
    formatted_results = []
    for index, scores in enumerate(zip(*all_scores)):
        sentence_confidence = sum(scores) / len(scores)
        formatted_results.append(f"{sentences[index].strip()} ({sentence_confidence})")
    
    formatted_output = " ".join(formatted_results)
    return formatted_output


def run(LLM, query, sample_size=5):
    sampled = sample_answer(LLM, query, sample_size + 1)
    answer = sampled[0]
    proofs = sampled[1:]
    sentences = sent_tokenize(answer)

    all_scores = []
    for proof in proofs:
        scores = check_score(LLM, proof, sentences)
        all_scores.append(scores)
    avg_confidence = sum([sum(scores) / len(scores) for scores in zip(*all_scores)]) / len(sentences)
    
    final_content = format_output(sentences, all_scores, avg_confidence)
    print(final_content+'\n'+str(avg_confidence))
    return final_content,avg_confidence