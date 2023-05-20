import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

cosine_similarity_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5', device='cpu')
cosine_similarity_embeddings = np.load('msmarco-distilbert-cos-v5_emb.npy')
cosine_similarity_index = faiss.IndexFlatIP(768)
cosine_similarity_index.add(cosine_similarity_embeddings)

dotprod_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device='cpu')
dotprod_embeddings = np.load('msmarco-distilbert-base-tas-b_emb.npy')
dotprod_index = faiss.IndexFlatIP(768)
dotprod_index.add(dotprod_embeddings)

with open('processed_books_metadata.json') as f:
    metadata = json.load(f)

def semantic_search(query, model):
    if model == "msmarco-distilbert-cos-v5 (cosine similarity)":
        query_embedding = cosine_similarity_model.encode([query])
        D, I = cosine_similarity_index.search(query_embedding, 5)
    elif model == "msmarco-distilbert-base-tas-b (dot product)":
        query_embedding = dotprod_model.encode([query])
        D, I = dotprod_index.search(query_embedding, 5)
    D, I = list(D[0]), list(I[0])
    result_list = [(metadata[index]["Name"] + " by " + metadata[index]["Author"], float(distance)) 
         for index, distance in zip(I, D)]
    result = "".join([f"""### {idx+1}. {result[0]}: {result[1]:.2f}\n""" for idx, result in enumerate(result_list[:5])])
    result = f"""# Top 5 Results\n{result}"""
    return result


demo = gr.Interface(
    fn=semantic_search, 
    inputs=[gr.Textbox(label="Query"),
            gr.Radio(["msmarco-distilbert-cos-v5 (cosine similarity)", "msmarco-distilbert-base-tas-b (dot product)"], 
                     label="Model")], 
    outputs=gr.Markdown(label="Results"))
    # outputs=gr.Label(num_top_classes=5, label="Cosine Similarity"))

demo.launch()