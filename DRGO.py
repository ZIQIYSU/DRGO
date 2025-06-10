from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import sent_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
import hydra
from tqdm import tqdm
from utils.llm import LLM_CLS
from utils.retrieve import (
    run,
    retrieve_top5,
    retrieve_queries
)
import gzip
import numpy as np
import time
import json

@hydra.main(version_base=None, config_path="configs", config_name="DRGO")
def main(cfg: DictConfig) -> None:
    base_url = os.getenv('BASE_URL')
    api_key = os.getenv('API_KEY')

    sentence_embeddings = np.load(cfg.benchmark.vector_storage)
    sentences = []
    if 'arc' in cfg.benchmark.name.lower() or 'openbook' in cfg.benchmark.name.lower():
        sentences = [line.strip() for line in open(cfg.benchmark.corpus, 'r').readlines() if line.strip()]
    elif 'commonsense' in cfg.benchmark.name.lower() or 'piqa' in cfg.benchmark.name.lower() or 'fever' in cfg.benchmark.name.lower():
        for line in gzip.open(cfg.benchmark.corpus, 'rt', encoding='utf-8'):
            item = json.loads(line)
            title = item.get("title", "")
            text = item.get("text", "")
            sentences.append({"title": title, "text": text})
    else :
        raise ValueError(f"Unsupported benchmark name: {benchmark}")
    
    em_tokenizer = AutoTokenizer.from_pretrained(cfg.retriever_path)
    em_model = AutoModel.from_pretrained(cfg.retriever_path)
    em_model.eval()
    
    LLM = LLM_CLS(cfg.llm, base_url, api_key)
    file_name = f"{cfg.name}_{cfg.benchmark.name}_{cfg.llm}_{str(cfg.threshold).replace('.', '_')}_{str(cfg.benchmark.sample_size)}.jsonl"
    file_path = os.path.join(cfg.save_path, file_name)
    benchmark = cfg.benchmark.name.lower()
    results = []
    lines = open(cfg.benchmark.dataset.data_path,'r').readlines()
    with open(file_path,'w') as output_file:
        for line in tqdm(lines, desc="Processing", unit="line"):
            record = json.loads(line)
            if 'arc' in benchmark.lower():
                question = record['question']
                choices = record['choices']
                choices_text = ' '.join([f"({label}) {text}" for text, label in zip(choices['text'], choices['label'])])
                result = f"{question} {choices_text}"
                start_time = time.time()
                answer, score=run(LLM = LLM, query=f'''Use what you already know to choose the best option for the question. 
                        Question: {result}''', sample_size=cfg.benchmark.sample_size)
                end_time = time.time()
                excute_time = end_time - start_time
            elif 'commonsense' in benchmark.lower():
                question = record['question']
                choices = record['choices']
                choices_text = ' '.join([f"({label}) {text}" for text, label in zip(choices['text'], choices['label'])])
                result = f"{question} {choices_text}"
                start_time = time.time()
                answer, score=run(LLM = LLM, query=f'''Please choose the best option for the question. 
                        Question: {result}''', sample_size=cfg.benchmark.sample_size)
                end_time = time.time()
                excute_time = end_time - start_time
            elif 'openbook' in benchmark.lower():
                question = record['question']['stem']
                choices = record['question']['choices']
                choices_text = ' '.join([f"({choice['label']}) {choice['text']}" for choice in choices])
                result = f"{question} {choices_text}"
                start_time = time.time()
                answer, score=run(LLM = LLM, query=f'''Use what you already know to choose the best option for the question. 
                        Question: {result}
                        Explanation:''', sample_size=cfg.benchmark.sample_size)
                end_time = time.time()
                excute_time = end_time - start_time
            elif 'piqa' in benchmark.lower():
                question = record['goal']
                result = f"{record['goal']}? (A) {record['sol1']}. (B) {record['sol2']}."
                start_time = time.time()
                answer, score=run(LLM = LLM, query=f'''Use what you already know to choose the best option for the question. 
                        Question: {result}
                        Explanation:''', sample_size=cfg.benchmark.sample_size)
                end_time = time.time()
                excute_time = end_time - start_time
            elif 'fever' in benchmark.lower():
                question = record['claim']
                result = record['claim']
                start_time = time.time()
                answer, score=run(LLM = LLM, query=f'''Use what you already know to verify a fact. Give your choice in ['SUPPORTS','REFUTES','NOT ENOUGH INFO'] 
                        Fact: {result}
                        Explanation:''', sample_size=cfg.benchmark.sample_size)
                end_time = time.time()
                excute_time = end_time - start_time
            else:
                raise ValueError(f"Unsupported benchmark name: {benchmark}")
            
            if score>=cfg.threshold:
                if 'piqa' in benchmark.lower():
                    output_record = {
                        "result": result,
                        "llm_answer": answer,
                        "true_answer": record['answer'],
                        "score": score,
                        "run_time": excute_time
                    }
                elif 'fever' in benchmark.lower():
                    output_record = {
                        "result": result,
                        "llm_answer": answer,
                        "true_answer": record['label'],
                        "score": score,
                        "run_time": excute_time
                    }
                else:
                    output_record = {
                        "result": result,
                        "llm_answer": answer,
                        "true_answer": record['answerKey'],
                        "score": score,
                        "run_time": excute_time
                    }
            else:
                # Extract sentences and their corresponding scores from the answer
                sentences_with_scores = re.findall(r'([^.]*\.)\s*\(([\d\.]+)\)', answer)  # Match sentences and scores
                
                # Filter out sentences below the threshold
                low_confidence_sentences = [sentence for sentence, score in sentences_with_scores if float(score) < cfg.threshold]
                
                # If there are low confidence sentences, combine them into a new text
                if low_confidence_sentences:
                    low_confidence_text = " ".join(low_confidence_sentences)
                else:
                    low_confidence_text = ""

                # Extracting factual statements from low confidence sentences as a whole
                if low_confidence_text:
                    fact_statements = LLM(f"{low_confidence_text}. Use the [STA] and [END] tags to list the specific factual statements contained in the above answers. Be complete and don't leave out any facts. Provide each claim as a separate sentence. For example, [STA] xxx [END]. [STA] xxx [END].")
                    sentence=str(fact_statements)
                    print(sentence)
                    start_time = time.time()
                    query_retrieve = retrieve_top5(benchmark, em_model, em_tokenizer, result, sentence_embeddings, sentences)
                    fact_retrieve = retrieve_queries(benchmark, em_model, em_tokenizer, sentence, sentence_embeddings, sentences)
                    end_time = time.time()
                    retrieve_time = end_time - start_time
                    # Merge all search results
                    all_retrieve = query_retrieve + fact_retrieve
                    # Remove duplicates and maintain order
                    if 'arc' in cfg.benchmark.name.lower() or 'openbook' in cfg.benchmark.name.lower():
                        all_retrieve = list(dict.fromkeys(all_retrieve))
                    elif 'commonsense' in cfg.benchmark.name.lower() or 'piqa' in cfg.benchmark.name.lower() or 'fever' in cfg.benchmark.name.lower():
                        all_retrieve = list({tuple(sorted(d.items())):d for d in all_retrieve}.values())
                    
                else:
                    start_time = time.time()
                    all_retrieve = retrieve_top5(benchmark, em_model, em_tokenizer, question, sentence_embeddings, sentences)
                    end_time = time.time()
                    retrieve_time = end_time - start_time
                    sentence=[]
                # Combine the problem and retrieved content into a new prompt statement
                if 'fever' in benchmark.lower():
                    content_with_retrieves = f"""Please verify a fact based on the additional retrieved information.Give your choice in ['SUPPORTS','REFUTES','NOT ENOUGH INFO'] 
                    Fact: {result}\n                
                    Retrieved Information: {all_retrieve}"""
                else:
                    content_with_retrieves = f"""Please answer the question based on the additional retrieved information.
                    Question: {result}\n                
                    Retrieved Information: {all_retrieve}"""
                final_answer=LLM(content_with_retrieves)
                if 'piqa' in benchmark.lower():
                    output_record = {
                        "result": result,
                        "Original answer":answer,
                        "sentence":sentence,
                        "retrieve": all_retrieve,
                        "llm_answer": final_answer,
                        "true_answer": record['answer'],
                        "score": score,
                        "run_time":excute_time,
                        "retrieve_time": retrieve_time
                    }
                if 'fever' in benchmark.lower():
                    output_record = {
                        "result": result,
                        "Original answer":answer,
                        "sentence":sentence,
                        "retrieve": all_retrieve,
                        "llm_answer": final_answer,
                        "true_answer": record['label'],
                        "score": score,
                        "run_time":excute_time,
                        "retrieve_time": retrieve_time
                    }
                else:
                    output_record = {
                        "result": result,
                        "Original answer":answer,
                        "sentence":sentence,
                        "retrieve": all_retrieve,
                        "llm_answer": final_answer,
                        "true_answer": record['answerKey'],
                        "score": score,
                        "run_time":excute_time,
                        "retrieve_time": retrieve_time
                    }
                
            print(output_record)
            
            output_file.write(json.dumps(output_record) + '\n')
            print("已存入")
        
if __name__ == '__main__':
    main()