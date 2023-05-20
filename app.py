import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

cache = {}

with open('processed_books_metadata.json') as f:
    metadata = json.load(f)

def semantic_search(query, model_name):
    if model_name not in cache:
        model = SentenceTransformer(f'sentence-transformers/{model_name}', device='cpu')
        embeddings = np.load(f'{model_name}_emb.npy')
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        cache[model_name] = (model, index)
    else:
        model, index = cache[model_name]
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, 5)
    # print(D, I)
    D, I = list(D[0]), list(I[0])
    result_list = [(metadata[index]["Name"] + " by " + metadata[index]["Author"], float(distance)) 
         for index, distance in zip(I, D)]
    # print(result_list)
    result = "".join([f"""### {idx+1}. {result[0]}: {result[1]:.2f}\n""" for idx, result in enumerate(result_list[:5])])
    result = f"""# Top 5 Results\n{result}"""
    return result


demo = gr.Interface(
    fn=semantic_search, 
    inputs=[gr.Textbox(label="Query", value="Phylosophical novel about 2d and 3d worlds"),
            gr.Dropdown(["msmarco-distilbert-cos-v5", "msmarco-MiniLM-L6-cos-v5", 
                         "msmarco-MiniLM-L12-cos-v5", "msmarco-distilbert-base-tas-b", 
                         "msmarco-distilbert-dot-v5", "msmarco-bert-base-dot-v5"], 
                     label="Model", multiselect=False, value="msmarco-distilbert-base-tas-b")], 
    outputs=gr.Markdown(label="Results"))
    # outputs=gr.Label(num_top_classes=5, label="Cosine Similarity"))

demo.launch()