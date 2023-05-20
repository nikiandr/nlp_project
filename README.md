# Asymmetric semantic search using document contextual embeddings on long documents

## Overview
Asymmetric semantic search is a task of matching short prompts and long texts based on semantic meaning. This repository contains project done as part of NLP course in University of Tartu. 

Main idea of the project was to collect dataset of books, preprocess it, transform them into embeddings using Sentence-BERT and then use embedding of the input prompt 

Report about it could be found [here](./NLP_Project_Report.pdf).

You can access demo for the project on [Hugginface Spaces](https://huggingface.co/spaces/nikiandr/assym_sem_search).

## Dataset
Dataset contains 78 preprocessed books from Project Guttenberg library. Dataset could be found [here](./data/processed_books.json), additional metadata used for evaluation - [here](./data/processed_books_metadata.json). More details on dataset could be found in the [report](./NLP_Project_Report.pdf).

Code for preprocessing original book files into structured dataset can be found [here](./preprocess.ipynb).

## Embeddings generation

For generating appropriate embeddings for the task 6 Sentence-BERT pretrained models were tested out (see [report](./NLP_Project_Report.pdf) for more details and evaluation). Code for generating embeddings and testing them out coudl be found [here](./embeddings.ipynb).

## Evaluation

For evaluation descriptions of the books from metadata were used as queries. Metrics used were top-1, top-5 and top-10 accuracies.