import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="AI PDF Chat", layout="wide")
    st.title("📄 AI PDF Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input("Enter Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory.clear()

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file and api_key:
        if st.session_state.vectorstore is None:
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_file)
                st.success("PDF processed and indexed!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF"):
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar.")
        elif st.session_state.vectorstore is None:
            st.error("Please upload a PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=st.session_state.memory
                )
                response = qa_chain.invoke({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
