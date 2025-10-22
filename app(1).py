import streamlit as st
import boto3
from io import BytesIO
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from PyPDF2 import PdfReader
import docx

# Load environment variables
load_dotenv()

# AWS Bedrock Configuration
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

cohere_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

class RAGPipeline:
    def __init__(self):
        self.embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id=os.getenv('EMBEDDING_MODEL', 'amazon.titan-embed-text-v2:0')
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
            length_function=len,
        )
        self.vector_store = None
        
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded document"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_type == 'docx':
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        elif file_type == 'txt':
            return uploaded_file.read().decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def create_vector_store(self, text: str):
        """Create FAISS vector store from text"""
        chunks = self.text_splitter.split_text(text)
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        return chunks
    
    def retrieve_documents(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k documents without reranking"""
        if self.vector_store is None:
            return []
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in docs_with_scores]
    
    
    def rerank_documents(self, query: str, documents: List[str], top_n: int) -> List[Dict]:
        """Rerank documents using Cohere Rerank 3.5 via bedrock-runtime.invoke_model"""
        # Bedrock Cohere Rerank expects raw strings, not {"text": "..."} objects
        request_body = {
            "query": query,
            "documents": documents,            # list[str]
            "top_n": int(top_n),
            "api_version": 2                   # REQUIRED for Cohere Rerank via bedrock-runtime
        }
    
        response = cohere_runtime.invoke_model(
            modelId=os.getenv('RERANK_MODEL', 'cohere.rerank-v3-5:0'),
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
    
        response_body = json.loads(response['body'].read())
        results = response_body.get('results', [])
    
        # Map back to original document text
        formatted_results = []
        for r in results:
            idx = r.get('index', 0)
            formatted_results.append({
                'relevance_score': r.get('relevance_score', 0.0),
                'index': idx,
                'document': {'text': documents[idx]}
            })
    
        return formatted_results

    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Claude Sonnet 3.7"""
        prompt = f"""Based on the following context, please answer the question. Be specific and cite information from the context.

Context:
{context}

Question: {query}

Answer:"""
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(os.getenv('MAX_TOKENS', 2000)),
            "temperature": float(os.getenv('TEMPERATURE', 0.1)),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=os.getenv('LLM_MODEL', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'),
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']


def main():
    st.set_page_config(page_title="RAG with Cohere Rerank Comparison", layout="wide")
    
    st.title("🔍 RAG Application: With vs Without Cohere Rerank 3.5")
    st.markdown("Compare retrieval quality with and without Cohere Rerank 3.5 model")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Extract text
                    text = st.session_state.rag_pipeline.extract_text_from_file(uploaded_file)
                    
                    # Create vector store
                    chunks = st.session_state.rag_pipeline.create_vector_store(text)
                    st.session_state.chunks = chunks
                    st.session_state.document_processed = True
                    
                    st.success(f"✅ Document processed! Created {len(chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        if st.session_state.document_processed:
            st.info(f"📊 Total chunks: {len(st.session_state.chunks)}")
            
            with st.expander("View Document Chunks"):
                for i, chunk in enumerate(st.session_state.chunks):
                    st.text_area(
                        f"Chunk {i+1}",
                        chunk,
                        height=100,
                        key=f"chunk_{i}"
                    )
    
    # Main content area
    if st.session_state.document_processed:
        st.header("💬 Ask Questions")
        query = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            retrieve_k = st.slider("Initial retrieval count (k)", 5, 20, 10)
        
        with col2:
            rerank_top_n = st.slider("Reranked top documents", 3, 10, 5)
        
        if st.button("🚀 Compare Responses", type="primary"):
            if query:
                with st.spinner("Processing your query..."):
                    rag = st.session_state.rag_pipeline
                    
                    # ===== WITHOUT RERANK =====
                    st.markdown("---")
                    st.subheader("📌 WITHOUT Cohere Rerank")
                    
                    # Retrieve documents
                    retrieved_docs = rag.retrieve_documents(query, k=retrieve_k)
                    
                    # Display retrieved chunks
                    with st.expander(f"📚 Top {rerank_top_n} Retrieved Chunks (Vector Similarity)", expanded=True):
                        for i, (doc, score) in enumerate(retrieved_docs[:rerank_top_n]):
                            st.markdown(f"**Chunk {i+1}** - Similarity Score: `{score:.4f}`")
                            st.text_area(
                                f"Content_{i}",
                                doc,
                                height=150,
                                key=f"no_rerank_{i}",
                                label_visibility="collapsed"
                            )
                            st.markdown("---")
                    
                    # Generate response
                    context_without_rerank = "\n\n".join([doc for doc, _ in retrieved_docs[:rerank_top_n]])
                    response_without_rerank = rag.generate_response(query, context_without_rerank)
                    
                    st.markdown("### 🤖 Generated Response")
                    st.info(response_without_rerank)
                    
                    st.markdown("**Chunks Used for Response:**")
                    st.write(f"Top {rerank_top_n} chunks based on vector similarity scores")
                    
                    # ===== WITH RERANK =====
                    st.markdown("---")
                    st.subheader("⭐ WITH Cohere Rerank 3.5")
                    
                    # Retrieve and rerank
                    retrieved_texts = [doc for doc, _ in retrieved_docs]
                    reranked_results = rag.rerank_documents(query, retrieved_texts, top_n=rerank_top_n)
                    
                    # Display reranked chunks
                    with st.expander(f"🎯 Top {len(reranked_results)} Reranked Chunks", expanded=True):
                        for i, result in enumerate(reranked_results):
                            relevance_score = result.get('relevance_score', 0)
                            doc_text = result.get('document', {}).get('text', '')
                            
                            # Color code based on relevance
                            if relevance_score > 0.7:
                                color = "🟢"
                            elif relevance_score > 0.4:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.markdown(f"{color} **Chunk {i+1}** - Relevance Score: `{relevance_score:.4f}`")
                            st.text_area(
                                f"Reranked_Content_{i}",
                                doc_text,
                                height=150,
                                key=f"rerank_{i}",
                                label_visibility="collapsed"
                            )
                            
                            # Explain why this chunk was selected
                            st.markdown("**Why this chunk?**")
                            if relevance_score > 0.7:
                                st.success(f"High relevance ({relevance_score:.2f}): Strong semantic match with the query. Contains key information directly related to the question.")
                            elif relevance_score > 0.4:
                                st.warning(f"Medium relevance ({relevance_score:.2f}): Moderate semantic match. Contains related but not primary information.")
                            else:
                                st.error(f"Lower relevance ({relevance_score:.2f}): Weak semantic match. Included for context but may be tangentially related.")
                            
                            st.markdown("---")
                    
                    # Generate response with reranked context
                    context_with_rerank = "\n\n".join([
                        result.get('document', {}).get('text', '') 
                        for result in reranked_results
                    ])
                    response_with_rerank = rag.generate_response(query, context_with_rerank)
                    
                    st.markdown("### 🤖 Generated Response")
                    st.success(response_with_rerank)
                    
                    st.markdown("**Why these chunks were selected:**")
                    st.write("Cohere Rerank 3.5 uses advanced semantic understanding to identify the most relevant chunks. The model analyzes:")
                    st.markdown("""
                    - **Semantic relevance**: Deep understanding of query intent vs chunk content
                    - **Contextual meaning**: Beyond keyword matching to conceptual alignment
                    - **Information density**: Prioritizes chunks with higher answer potential
                    - **Query-document relationship**: Cross-attention between query and documents
                    """)
                    
                    # ===== COMPARISON SUMMARY =====
                    st.markdown("---")
                    st.subheader("📊 Comparison Summary")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Without Rerank**")
                        st.markdown(f"- Method: Vector similarity (cosine)")
                        st.markdown(f"- Chunks used: {rerank_top_n}")
                        st.markdown(f"- Selection: Top-k by embedding distance")
                        
                    with col_b:
                        st.markdown("**With Cohere Rerank 3.5**")
                        st.markdown(f"- Method: Semantic reranking")
                        st.markdown(f"- Chunks used: {len(reranked_results)}")
                        st.markdown(f"- Selection: Relevance-scored by Cohere")
                        if reranked_results:
                            avg_score = sum(r.get('relevance_score', 0) for r in reranked_results) / len(reranked_results)
                            st.markdown(f"- Avg relevance: {avg_score:.4f}")
            else:
                st.warning("Please enter a question!")
    else:
        st.info("👈 Please upload and process a document to begin")
        
        # Display sample use case
        st.markdown("---")
        st.subheader("📖 How it works")
        st.markdown("""
        1. **Upload Document**: Support for PDF, DOCX, and TXT files
        2. **Embedding**: Documents are chunked and embedded using AWS Titan
        3. **Vector Storage**: Chunks stored in FAISS vector database
        4. **Query Processing**: 
           - **Without Rerank**: Direct vector similarity search
           - **With Rerank**: Cohere Rerank 3.5 reorders results by semantic relevance
        5. **Response Generation**: Claude Sonnet 3.7 generates answers using retrieved context
        6. **Comparison**: Side-by-side view of both approaches with relevance scores
        """)


if __name__ == "__main__":
    main()