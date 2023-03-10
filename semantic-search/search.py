"""
#needed for semantic search
pip install torch
pip install transformers
pip install sentence-transformers

#only used in this demo
pip install pandas
pip install easyrepl


download dataset from https://www.kaggle.com/datasets/benhamner/nips-papers
extract into data/ folder (mainly just want papers.csv)
"""
from __future__ import annotations
import torch
from transformers import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import glob
import requests


# silence transformers logging. by default, mpnet prints out some unnecessary warnings
logging.set_verbosity_error()


def get_title(docid):
    resp = requests.get(f"https://xdd.wisc.edu/api/articles?docid={docid}")
    return resp.json()['success']['data'][0]['title']

def get_data(size=400) -> pd.DataFrame:
    #read in data, and remove rows with missing abstracts
    input_files = glob.glob("data/*.txt")
    data = []
    with open("doclist", "w") as fout:
        fout.write("\n".join(input_files))
    for f in input_files:
        docid = f.replace("data/", "").replace(".txt", "")
        text = open(f).read()[:5000]
        title = get_title(docid)
        data.append({"docid" : docid, "title" : title, "paper_text": text, "abstract": text})

#    df = pd.read_csv('data/papers.csv')
    df = pd.DataFrame(data)
#    df = df[df['abstract'] != 'Abstract Missing']
    df.reset_index(inplace=True, drop=True)

    #take a random subset of the data since it all probably won't fit in memory
#    df = df.sample(size, random_state=42)
#    df.reset_index(inplace=True, drop=True)

    return df


def get_title_abstract_data(size=400) -> list[str]:
    """Create a list of strings consisting of the title and abstract of each paper"""
    df = get_data(size)
#    data = (df['title'] + '\n' + df['abstract']).tolist()
    data = (df['title'] + '\n' + df['paper_text']).tolist()
    return data, df['docid'].tolist()



# from transformers import AutoTokenizer, AutoModel
# class SpecterEmbedder:
#     def __init__(self, try_cuda=True):
#         # load model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
#         self.model = AutoModel.from_pretrained('allenai/specter')

#         # move model to GPU if available
#         if try_cuda and torch.cuda.is_available():
#             self.model = self.model.cuda()

#         # store whether on GPU or CPU
#         self.device = next(self.model.parameters()).device

#     def embed(self, texts:list[str]):
#         # important to use no_grad otherwise it uses way too much memory
#         with torch.no_grad():
#             inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
#             result = self.model(**inputs)

#             embeddings = result.last_hidden_state[:, 0, :]

#             return embeddings


class MPNetEmbedder:
    def __init__(self, try_cuda=True):
        self.model = SentenceTransformer('all-mpnet-base-v2')

        # move model to GPU
        if try_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

        # save the device
        self.device = next(self.model.parameters()).device


    def embed(self, texts:list[str]):
        with torch.no_grad():
            embeddings = self.model.encode(texts, show_progress_bar=True, device=self.device, convert_to_tensor=True)
            return embeddings




def main():
    from easyrepl import REPL
    import pickle

    # instantiate the embedding model.
    # Note that Specter performs much worse than MPNet on this task, and probably shouldn't be used
    embedder = MPNetEmbedder()

    #embed the corpus of data. Elements are the title+abstract of each paper
    corpus, docids = get_title_abstract_data()
    corpus_embeddings = embedder.embed(corpus)
    pickle.dump(corpus_embeddings, open("output/embeddings.pkl", "wb"))
    pickle.dump(docids, open("output/docids.pkl", "wb"))
    pickle.dump(corpus, open("output/corpus.pkl", "wb"))


    #number of results to display
    top_k = 5

    print('Enter a query to search the corpus. ctrl+d to exit')
    for query in REPL():
        query_embedding = embedder.embed([query])

        # compute similarity scores
        scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings, dim=1)

        #get the indices of the top_k scores
        top_results = torch.topk(scores, k=top_k).indices.tolist()

        #print the results
        for idx in top_results:
            print()
            print(f"Match Score: {scores[idx]}")
            print(f"docid: {docids[idx]}, Text: {corpus[idx]}")
            print()

        print('-------------------------------------------------')



if __name__ == '__main__':
    main()
