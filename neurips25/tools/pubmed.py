'''
PubMed search using Biopython api which queries the NCBI Entrez API (PubMed) for a given search phrase

code inspired from https://github.com/google-gemini/gemma-cookbook/blob/main/TxGemma/%5BTxGemma%5DAgentic_Demo_with_Hugging_Face.ipynb
'''

import os
import re
from Bio import Medline, Entrez
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if os.getenv("DRUGBANK_USERNAME") is None:
    raise ValueError("Please set the DRUGBANK_USERNAME environment variable. (also used for pubmed search via Entrez)")
Entrez.email = os.getenv("DRUGBANK_USERNAME")

def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code

class PubMed:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')

    def rerank(self, query, retrieved_docs):
        tokenizer = self.tokenizer
        model = self.model
        model.eval()

        pairs = [[query, doc] for doc in retrieved_docs]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        sorted_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indexes



    def query(self, search_text : str = None, pmid : int = None, k : int = 3, k_before_rerank : int = 30) -> str:
        """
        Search PubMed for articles related to a given search term or fetch a specific article by its PubMed ID.

        Args:
            - search_text (str): The search term to query PubMed.
            - pmid (int): The PubMed ID of a specific article to fetch.
            - k (int): The number of top results to return.
        Returns:
            - str: A formatted string containing top-k search results or the article details.
        """
        assert search_text is not None or pmid is not None
        pmids = list()
        if search_text is not None:
            handle = Entrez.esearch(db="pubmed", sort="relevance", term=search_text, retmax=k_before_rerank)
            record = Entrez.read(handle)
        elif pmid is not None:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            record = handle.read()
            return record
        pmids = record.get("IdList", [])
        handle.close()

        if not pmids:
            return f"No PubMed articles found for '{search_text}' Please try a simpler search query."

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
        records = list(Medline.parse(fetch_handle))
        fetch_handle.close()

        if pmid is None and search_text is not None: # if the query is a search term we rerank the results for better relevance
            retrieved_docs = []
            for i, record in enumerate(records):
                # format for reranking
                record_str = f"Title: {record.get('TI', 'No title available')}\n"
                record_str += f"Journal: {record.get('JT', 'No journal info')}\n"
                record_str += f"Publication Date: {record.get('DP', 'No date info')}\n"
                record_str += f"Abstract: {record.get('AB', 'No abstract available')}\n"
                retrieved_docs.append(record_str)
            # rerank the retrieved docs
            sorted_indexes = self.rerank(search_text, retrieved_docs)
            # sort the records based on the reranked indexes
            records = [records[i] for i in sorted_indexes[:k]]

        result_str = f"=== PubMed Search Results for: '{search_text}' ===\n"
        for i, record in enumerate(records, start=1):
            pmid = record.get("PMID", "N/A")
            title = record.get("TI", "No title available")
            abstract = record.get("AB", "No abstract available")
            journal = record.get("JT", "No journal info")
            pub_date = record.get("DP", "No date info")
            authors = record.get("AU", [])
            authors_str = ", ".join(authors[:3])
            result_str += (
                f"\n--- Article #{i} ---\n"
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Authors: {authors_str}\n"
                f"Journal: {journal}\n"
                f"Publication Date: {pub_date}\n"
                f"Abstract: {abstract}\n")
        return f"Query: {search_text}\nResults: {result_str}"
    

def search_pubmed(search_text: str = None, pmid: int = None, k: int = 3) -> str:
    """
    Search PubMed for articles related to a given search term or fetch a specific article by its PubMed ID.

    Args:
        search_text: (str) The search term to query PubMed.
        pmid: (int) The PubMed ID of a specific article to fetch.
        k: (int) The number of top results to return.
    Returns:
        str: A formatted string containing top-k search results or the article details.
    """
    pubmed = PubMed()
    return pubmed.query(search_text=search_text, pmid=pmid, k=k)

if __name__ == "__main__":
    pubmed = PubMed()

    msk_patient = """tumor, clinical group 4, distant metastases/systemic disease, adenocarcinoma, nos, lung and bronchus
    non-small cell lung cancer, lung adenocarcinoma, hugo symbol TP53, somatic mutation status"""
    msk_patient = "tp53 lung adenocarcinoma somatic mutation"
    res = pubmed.query(search_text=msk_patient)
    print(res)

    # article_id = 3139862
    # res = pubmed.query(pmid=article_id)
    # print(res)
