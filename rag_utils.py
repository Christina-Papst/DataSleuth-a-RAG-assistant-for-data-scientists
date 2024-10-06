# This file contains the functions for the rag chatbot


# import the libraries
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import os
from rank_bm25 import BM25Plus
from pinecone_text.sparse import BM25Encoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus





def store_knowledgebase(model_name, corpus, index):
    """
    Store embeddings of documents from the corpus into the Pinecone index.

    Args:
        model_name (str): The name or path of the pre-trained language model to load for generating embeddings.
        corpus (list of str): A list of documents (text strings) to process and store in the index.
        index (pinecone.Index): The Pinecone index where the document embeddings will be stored.

    Returns:
        None
    """

    # initialize your llm
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Process each document in the corpus
    for i, doc in enumerate(corpus):
        unique_id = f"doc-{i}"  # Generate a unique ID for each document
        embedding = generate_embeddings(doc, tokenizer, model)
        store_embeddings(embedding, index, unique_id, doc)



def generate_embeddings(text, tokenizer, model):
    """
    Generate embeddings for a given text using a T5 model.

    Args:
        text (str): The input text.
        tokenizer (transformers.T5Tokenizer): The T5 tokenizer to use for encoding.
        model (transformers.T5ForConditionalGeneration): The T5 model to use for generating embeddings.

    Returns:
        numpy.ndarray: The generated embeddings.
    """

    # Decode the token IDs back to a string if necessary
    if isinstance(text, list):
        text = tokenizer.decode(text, skip_special_tokens=True)

    encoding = tokenizer.encode_plus(
        text,
        padding="longest",
        max_length=512,  # Adjust max_source_length as needed
        truncation=True,
        return_tensors="pt"
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # Use a dummy target sequence with the padding token
    dummy_target_sequence = tokenizer.pad_token_id

    # Create a target encoding with the dummy target sequence
    target_encoding = tokenizer.encode_plus(
        str(dummy_target_sequence),  # Convert the integer to a string
        padding="longest",
        max_length=1,  # Set max_target_length to 1 for a single token
        truncation=True,
        return_tensors="pt"
    )
    labels = target_encoding.input_ids

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=labels)
        embeddings = outputs.encoder_last_hidden_state[:, 0, :].squeeze()

    embeddings_np = embeddings.numpy()

    if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
        raise ValueError("Embeddings contain NaN or infinite values")

    return embeddings_np


def store_embeddings(embeddings, index, id, text):
    """
    Store the generated embeddings in the Pinecone database.

    Args:
        embeddings (numpy.ndarray): The generated embeddings.
        index (pinecone.Index): The Pinecone index where embeddings will be stored.
        id (str): The unique identifier for the text entry.
        text (str): The text content associated with the embeddings.

    Returns:
        None
    """
    # Convert embeddings to list
    embeddings_list = embeddings.tolist()

    # Create metadata with string values
    metadata = {
        "id": str(id),
        "text": text  # Use the actual text here
    }

    # Store the embeddings with metadata
    index.upsert([(id, embeddings_list, metadata)])


def clean_text(textfile):
    """
    Clean the text content by removing or replacing specific characters and patterns.

    Args:
        textfile (str): The text content to be cleaned.

    Returns:
        str: The cleaned text content.

    """
    textfile = re.sub("\n", " ", textfile) # replaces line breaks with white space
    textfile = re.sub("\t", " ", textfile) # replaces tabs with white space
    textfile = re.sub(r"\[\d+\]", " ", textfile) # removes the Wikipedia citation brackets
    textfile = re.sub(' +', ' ', textfile) # removes extra white space
    textfile = re.sub(':', ' ', textfile)
    textfile = re.sub(r'\?', ' ', textfile)  # escapes the question mark
    textfile = re.sub(r'\.', ' ', textfile)  # escapes the period
    textfile = re.sub('-', ' ', textfile)
    return textfile


def create_corpus(directory):
    """
    Create a corpus by reading and cleaning text files from a specified directory.

    Args:
        directory (str): The path to the directory containing .txt files.

    Returns:
        list: A list of cleaned text content from each .txt file in the directory.

    """
    corpus = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            # Open and read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # Clean the text content
                cleaned_content = clean_text(content)
                corpus.append(cleaned_content)
    return corpus


def process_corpus(corpus, model_name, index):
    """
    Process each document in the corpus to generate and store embeddings.

    Args:
        corpus (list): List of documents to process.
        model_name (str): The name or path of the pre-trained model to load.
        index: Index to store the embeddings.

    Returns:
        None
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    for i, doc in enumerate(corpus):
        unique_id = f"doc-{i}"  # Generate a unique ID for each document
        embedding = generate_embeddings(doc, tokenizer, model)
        store_embeddings(embedding, index, unique_id, doc)





def chunk_document(model_name, text, chunk_size, overlap):
    """
    Chunks a document into smaller segments.

    Args:
        text (str): The input text.
        chunk_size (int): The desired size of each chunk.
        overlap (int): The number of tokens to overlap between chunks.

    Returns:
        list: A list of chunks.
    """

    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)

    # Create chunks with overlap
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)

    return chunks


def query_pinecone(query, model_name, index, corpus, top_k, chunk_size, overlap, bm25_weight, semantic_search_weight, ranker):
    """
    Query the Pinecone index for the most similar embeddings, combining BM25 and semantic search with reranking.

    Args:
        query (str): The input text to be queried.
        model_name (str): The name or path of the pre-trained model to load.
        index (pinecone.Index): The Pinecone index to query.
        corpus (list): A list of documents to search through.
        top_k (int): The number of top results to return.
        chunk_size (int): The size of each chunk for processing.
        overlap (int): The overlap between chunks.
        bm25_weight (float, optional): The weight for BM25 scores.
        semantic_search_weight (float, optional): The weight for semantic search scores.
        ranker (Reranker): The reranker model to use for reranking.

    Returns:
        list: A list of the top_k most similar embeddings.
    """
    # initialize your llm
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Retrieve documents using BM25
    bm25 = BM25Plus(corpus)
    bm25_scores = bm25.get_scores(query.split())
    ranked_documents = [corpus[i] for i in bm25_scores.argsort()[::-1]]

    # Chunk the retrieved documents from BM25
    chunked_docs = []
    for doc in ranked_documents:
        doc_chunks = chunk_document(model_name, doc, chunk_size, overlap)
        chunked_docs.extend(doc_chunks)

    # Generate embeddings for each chunk from BM25
    embeddings = []
    for chunk in chunked_docs:
        chunk_text = tokenizer.decode(tokenizer.encode(str(chunk), add_special_tokens=True), skip_special_tokens=True)
        chunk_embedding = generate_embeddings(chunk_text, tokenizer, model)
        embeddings.append(chunk_embedding)

    # Query the Pinecone index and include metadata in the response
    query_embedding = generate_embeddings(query, tokenizer, model)
    query_dict = {"vector": query_embedding.tolist()}
    query_response = index.query(vector=query_dict["vector"], top_k=top_k, include_metadata=True)

    # Chunk the retrieved documents from the vector search
    chunked_semantic_docs = []
    for match in query_response['matches']:
        doc_text = match['metadata']['text']
        doc_chunks = chunk_document(model_name, doc_text, chunk_size, overlap)
        chunked_semantic_docs.extend(doc_chunks)

    # Generate embeddings for chunked semantic documents
    semantic_embeddings = []
    for chunk in chunked_semantic_docs:
        chunk_text = tokenizer.decode(tokenizer.encode(str(chunk), add_special_tokens=True), skip_special_tokens=True)
        chunk_embedding = generate_embeddings(chunk_text, tokenizer, model)
        semantic_embeddings.append(chunk_embedding)

    # Normalize BM25 and semantic scores before combining (optional)
    epsilon = 1e-10  # Small value to avoid division by zero
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    semantic_scores = [match['score'] for match in query_response['matches']]
    semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + epsilon)

    combined_scores = []
    for match, bm25_score, semantic_score in zip(query_response['matches'], bm25_scores, semantic_scores):
        combined_score = bm25_score * bm25_weight + semantic_score * semantic_search_weight
        combined_scores.append((match, combined_score))

    # Sort results based on combined scores
    sorted_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Extract the text of the relevant documents
    relevant_docs = [result[0]['metadata']['text'] for result in sorted_results[:top_k]]

    # Prepare documents for reranking
    doc_ids = [result[0]['id'] for result in sorted_results[:top_k]]

    # Rerank the retrieved documents
    ranked_results = ranker.rank(query, relevant_docs, doc_ids)

    # Sort the results based on the reranker's scores
    final_sorted_results = sorted(ranked_results.results, key=lambda x: x.score, reverse=True)

    return final_sorted_results[:top_k]


def generate_response(query, prompt, model_name, index, corpus, top_k, chunk_size, overlap, temperature, max_new_tokens, bm25_weight, semantic_search_weight, ranker):
    """
    Generate a response based on the query and prompt using the Pinecone index and a language model.

    Args:
        query (str): The input query text.
        prompt (str): The prompt to guide the response generation.
        model_name (str): The name or path of the pre-trained model to load.
        index (pinecone.Index): The Pinecone index to query.
        corpus (list): A list of documents to search through.
        top_k (int, optional): The number of top results to return from the Pinecone query.
        chunk_size (int, optional): The size of each chunk for processing.
        overlap (int, optional): The overlap between chunks.
        temperature (float, optional): The temperature parameter to control creativity.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
        bm25_weight (float, optional): The weight for BM25 scores.
        semantic_search_weight (float, optional): The weight for semantic search scores.
        ranker (Reranker): The reranker model to use for reranking.

    Returns:
        str: The generated response text.
    """
    # initialize your llm
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Query Pinecone to get relevant documents
    query_response = query_pinecone(query, model_name, index, corpus, top_k, chunk_size, overlap, bm25_weight, semantic_search_weight, ranker)

    # Debugging statements to inspect the query_response object
    #print(type(query_response))
    #print(query_response)

    # Extract the text of the relevant documents
    relevant_docs = [match.document.text for match in query_response if match is not None]

    # Combine the relevant documents into a single string
    combined_docs = " ".join(relevant_docs)

    # Generate the response using the language model
    inputs = tokenizer(prompt + combined_docs, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response