import pandas as pd
import streamlit as st
import DocumentClassifier
from pdfminer.pdfparser import PDFParser


st.title("FreeOpenUniversity: Document Classification Module")
st.sidebar.title("Setting")
upload_method = st.sidebar.selectbox("Upload file or input text?", ('File Uploader', 'Textbox'))
document_length = st.sidebar.selectbox("Paragraph, chapter or entire text book", ('Paragraph', 'Chapter', 'Text Book'))
if upload_method == 'File Uploader':
    document_type = st.sidebar.selectbox("Select file format", ('txt', 'pdf'))
    uploaded_file = st.file_uploader('Upload your document here', type=document_type)
    if document_type is 'txt':
        document_file = uploaded_file
    if document_type is 'pdf':
        document_file = PDFParser(uploaded_file)
else:
    document_file = st.text_area('Paste your text here')
# st.write(type(uploaded_file))

TFIDF_document_classifier = DocumentClassifier.TFIDFDocumentClassifier.load('./temp/TF_IDF_MOCK_CLF.joblib',
                                                                            vpath='./temp/TF_IDF_MOCK_V.joblib')
mapping = {0: 'business',
 1: 'entertainment',
 2: 'food',
 3: 'graphics',
 4: 'historical',
 5: 'medical',
 6: 'politics',
 7: 'space',
 8: 'sport',
 9: 'technologie'}

if st.button('Classify'):
    # st.write(TFIDF_document_classifier.predict([document_file])[0])
    prediction_result = TFIDF_document_classifier.predict([document_file])[0]

    st.write(mapping[int(prediction_result)])
