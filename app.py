import pandas as pd
import streamlit as st

st.title("FreeOpenUniversity: Document Classification Module")
st.sidebar.title("Setting")
upload_method = st.sidebar.selectbox("Upload file or input text?", ('File Uploader', 'Textbox'))
document_length = st.sidebar.selectbox("Paragraph, chapter or entire text book", ('Paragraph', 'Chapter', 'Text Book'))
if upload_method == 'File Uploader':
    document_type = st.sidebar.selectbox("Select file format", ('txt', 'pdf'))
    uploaded_file = st.file_uploader('Upload your document here', type=document_type)
else:
    document_file = st.text_area('Paste your text here')
# st.write(type(uploaded_file))
st.write('The upload document belongs to XXX category')
