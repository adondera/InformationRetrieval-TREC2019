from pyserini.search import SimpleSearcher
import pandas as pd

searcher = SimpleSearcher('indexes/lucene-index-msmarco-doc')
searcher.set_bm25(k1=4.46, b=0.82)

queries = pd.read_csv('queries.docdev.tsv', sep='\t').to_numpy()

counter = 0
with open('output.trec', 'w') as fout:
    for q in queries:
        counter += 1
        if counter % 10 == 0:
            print(counter)
        hits = searcher.search(q[1], k=100)
        for (idx, hit) in enumerate(hits):
            score = 1.0 / int(idx + 1)
            fout.write(f'{q[0]} Q0 {hit.docid} {idx} {score} anserini\n')


