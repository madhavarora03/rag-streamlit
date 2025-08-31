import base64
import gc
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load environment variables
load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = OpenAI()


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


# Page config
st.set_page_config(page_title="RAG Streaming Chat", layout="centered")

with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):
                    loader = PyPDFLoader(file_path=str(file_path))
                    docs = loader.load()

                    # Chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=400
                    )
                    split_docs = text_splitter.split_documents(docs)

                    # Vector Embeddings
                    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
                    vector_store = QdrantVectorStore.from_documents(
                        documents=split_docs,
                        url="http://localhost:6333",
                        collection_name="rag-pdf",
                        embedding=embedding_model,
                    )

                    # ✅ Store all relevant info
                    st.session_state.file_cache[file_key] = {
                        "vector_store": vector_store,
                        "docs": split_docs,
                        "embedding_model": embedding_model,
                        "file_path": file_path,
                    }
                else:
                    cache_entry = st.session_state.file_cache[file_key]
                    vector_store = cache_entry["vector_store"]
                    split_docs = cache_entry["docs"]
                    embedding_model = cache_entry["embedding_model"]
                    file_path = cache_entry["file_path"]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])
with col1:
    st.header("Chat with Docs using GPT-4.1")
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Always connect to existing vector DB
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_db = QdrantVectorStore.from_existing_collection(
                url="http://localhost:6333",
                collection_name="rag-pdf",
                embedding=embedding_model,
            )

            # Search top-k docs
            search_results = vector_db.similarity_search(query=prompt, k=4)

            # Context prep
            context = "\n\n\n".join(
                [
                    f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label','N/A')}\nFile Location: {result.metadata.get('source','N/A')}"
                    for result in search_results
                ]
            )

            SYSTEM_PROMPT = f"""
                You are a helpful AI Assistant who answers based only on the retrieved PDF context.
                Always cite the page number(s) and tell the user where to look in the PDF.

                Context:
                {context}
            """

            # Stream GPT response
            chat_completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
