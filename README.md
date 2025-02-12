Create and activate a virtual environemnt : python -m venv .venv source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install the required packages: pip install -r requirements.txt
Start the Ollama service and ensure the Deepseek model is available: ollama run deepseek-r1:1.5b
Run the Streamlit application: streamlit run streamlit.py
Access the application in your web browser at http://localhost:8501
Upload a PDF document using the file uploader
Ask questions about the document content using the chat input
Project Structure :
pdf-qa-system/
├── main.py                # Main application file
├── pdfs/                  # Directory for uploaded PDFs
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
