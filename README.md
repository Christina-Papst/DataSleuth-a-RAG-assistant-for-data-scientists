# DataSleuth: a RAG assistant for Data Scientists

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot designed to help data scientists query large amounts of unstructured text data. The chatbot runs within a Jupyter Notebook, making it easy to integrate into data science workflows.

## How It Works
The chatbot combines vector search and BM25 to provide the best possible answers:

#### Vector Search: 
Uses Pinecone to perform vector search, which finds the most semantically similar documents to the query. This method captures the contextual meaning of the query and the documents.

#### BM25: 
A traditional information retrieval algorithm that ranks documents based on the frequency of query terms appearing in them. BM25 is effective for keyword-based searches.

By combining these two methods, the chatbot leverages the strengths of both semantic understanding and keyword matching to deliver accurate and relevant responses.

## Features
- **Query large text datasets**: Efficiently search and retrieve information from a collection of text files.
- **Powered by T5 model**: Utilizes the T5 language model for generating responses.
- **Jupyter Notebook integration**: Designed to run within a Jupyter Notebook for seamless use by data scientists.

## Setup

### Prerequisites
- Python 3.x
- Jupyter Notebook
- API key for Pinecone database
- A folder containing `.txt` files you want to query

### Steps
-  Download the repository onto your computer
-  Create a folder that contains the texts you want to query as .txt files
-  Follow the instructions in the run_rag.ipynb file
-  Feel free to experiment with the different hyperparameters to optimize your results


## Limitations
#### Hardware constraints: 
The code was developed on a personal laptop, which may not be powerful enough to handle multiple large models simultaneously. For better performance, consider using a machine with higher computational power or cloud-based solutions.

#### Model limitations: 
The T5 model, while powerful, may not always provide perfect responses and can be resource-intensive.

#### Data dependency: 
The effectiveness of the chatbot heavily depends on the quality and relevance of the text data provided.
