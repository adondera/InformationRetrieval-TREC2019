from pyserini.search import SimpleSearcher
import pandas as pd
import time

queries = pd.read_csv('queries.docdev.tsv', sep='\t', names=['query_id', 'query'], dtype={"query_id": str})

searcher = SimpleSearcher('indexes/lucene-index-msmarco-doc')
searcher.set_bm25(k1=4.46, b=0.82)

counter = 0

start = time.time()

batch_size = 64
threads = 12
with open('output4.trec', 'w') as fout:
    i = 0
    while i < len(queries):
        batch = queries.iloc[i:min(i + batch_size, len(queries))]
        hits = searcher.batch_search(queries=batch['query'].to_list(), qids=batch['query_id'].to_list(), threads=8,
                                     k=100)
        for q_id, results in hits.items():
            for (idx, hit) in enumerate(results):
                score = 1.0 / int(idx + 1)
                fout.write(f'{q_id} Q0 {hit.docid} {idx} {score} anserini\n')
        i += batch_size
        print(f'{i} queries processed')
print(f'Took {time.time() - start}s to finish')
