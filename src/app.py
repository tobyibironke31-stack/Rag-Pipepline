import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from parser import load_pdf_with_pymupdf
from hybrid_search import HybridRetriever

load_dotenv()

# Initialize embeddings and LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")
Settings.llm = GoogleGenAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

rag_engine = None

def process_pdf(file):
    global rag_engine
    if file is None:
        return "Please upload a valid PDF."

    docs = load_pdf_with_pymupdf(file.name)
    index = VectorStoreIndex.from_documents(docs)

    nodes = list(index.docstore.docs.values())
    safe_top_k = min(2, max(1, len(nodes)))

    vector_retriever = index.as_retriever(similarity_top_k=safe_top_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=safe_top_k)

    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, top_k=safe_top_k)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=safe_top_k
    )

    rag_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm,
        node_postprocessors=[reranker],
    )
    return f"Indexed {len(docs)} pages successfully. You can now submit questions."

def handle_chat(message, history):
    if rag_engine is None:
        return history + [[message, "Error: Upload and process a PDF document first."]]
    response = rag_engine.query(message)
    return history + [[message, str(response)]]

with gr.Blocks(title="Hybrid RAG Document QA") as demo:
    gr.Markdown("# 📄 Hybrid RAG PDF Assistant\nDense + Sparse Retrieval with Cross-Encoder Reranking")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=400)
            user_input = gr.Textbox(placeholder="Ask a question about the PDF...", label="Question")
            with gr.Row():
                send_btn = gr.Button("📤 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")
        
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload Document", file_types=[".pdf"])
            status_box = gr.Textbox(label="Status", interactive=False)
            process_btn = gr.Button("🔄 Process Document")

    process_btn.click(process_pdf, inputs=pdf_input, outputs=status_box)
    send_btn.click(handle_chat, inputs=[user_input, chatbot], outputs=chatbot)
    user_input.submit(handle_chat, inputs=[user_input, chatbot], outputs=chatbot)
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
