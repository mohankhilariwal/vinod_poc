import streamlit as st
import main
import os

st.title("Chat with PDFs with Deepseek")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    file_path = main.upload_pdf(uploaded_file)  # Get the file path

    if file_path is None or not os.path.exists(file_path):  # Check file existence
        st.error("Error: File was not saved properly. Please try again.")
    else:
        st.success(f"File saved successfully at: {file_path}")

        try:
            #  Check if ChromaDB already exists
            if os.path.exists(main.chroma_db_directory):
                st.info("ℹ️ Using existing ChromaDB...")
                db = main.load_existing_chroma_db()  # Load existing DB
            else:
                st.info("⏳ Creating new ChromaDB...")
                db = main.create_vector_store(file_path)  # Create new vector store
                st.success("Vector store created successfully!")

            question = st.chat_input()
            if question:
                st.chat_message("user").write(question)
                related_documents = main.retrieve_docs(db, question)
                answer = main.question_pdf(question, related_documents)
                st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f" Error while processing PDF: {e}")  # ✅ Print error in UI
