# TREC 2019 deep learning track

* Download the dataset from https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#deep-learning-track-tasks
* Save the collection in `collections/msmarco-doc/msmarco-docs.trec.gz`
* Build the index using the following command:
* ```shell
  python -m pyserini.index.lucene --collection CleanTrecCollection 
                                --input collections/msmarco-doc
                                --index indexes/lucene-index-msmarco-doc
                                --generator DefaultLuceneDocumentGenerator
                                --threads 1
                                --storePositions
                                --storeDocvectors
                                --storeRaw
  ```

* In `coreIR.py` the script used for retrieving results using BM25 scoring can be found
* The `query_expansion.ipynb` notebook contains the code necessary for expanding queries according to cosine similarity
* The `prepare_data_l2r.ipynb` notebook calculates a set of features for each query-document pair
* The `lambdamart.ipynb` notebook contains the code that fits a LambdaMART model and runs it on the `dev` and `test` sets.
  * Similarly for the `logisitcregression.ipynb`
* Lastly `analysis.ipynb` contains some rough scripts used for error analysis.