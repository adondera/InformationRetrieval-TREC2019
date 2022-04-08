from pyserini.search import SimpleSearcher
import pandas as pd
import time

queries = pd.read_csv('test/msmarco-test2019-queries.tsv', sep='\t', names=['query_id', 'query'],
                      dtype={"query_id": str})

searcher = SimpleSearcher('indexes/lucene-index-msmarco-doc')
searcher.set_bm25(k1=4.46, b=0.82)
k = 100

batch_search = True

counter = 0

start = time.time()

batch_size = 64
threads = 12
with open(f'results/test-bm25-{k}.trec', 'w') as fout:
    i = 0
    if batch_search:
        while i < len(queries):
            batch = queries.iloc[i:min(i + batch_size, len(queries))]
            hits = searcher.batch_search(queries=batch['query'].to_list(), qids=batch['query_id'].to_list(), threads=8,
                                         k=k)
            for q_id, results in hits.items():
                for (idx, hit) in enumerate(results):
                    score = hit.score
                    fout.write(f'{q_id} Q0 {hit.docid} {idx} {score} anserini\n')
            i += batch_size
            print(f'{i} queries processed')
    else:
        for q in queries.to_numpy():
            counter += 1
            if counter % 10 == 0:
                print(counter)
            hits = searcher.search(q[1], k=k)
            for (idx, hit) in enumerate(hits):
                fout.write(f'{q[0]} Q0 {hit.docid} {idx} {hit.score} anserini\n')
print(f'Took {time.time() - start}s to finish')
