import pandas as pd
import streamlit as st
import DocumentClassifier
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import *
from pdfminer.layout import *
from pdfminer.image import ImageWriter
from pdfminer.converter import TextConverter

st.title("FreeOpenUniversity: Document Classification Module")
st.sidebar.title("Setting")
upload_method = st.sidebar.selectbox("Upload file or input text?", ('File Uploader', 'Textbox'))
document_length = st.sidebar.selectbox("Paragraph, chapter or entire text book", ('Paragraph', 'Chapter', 'Text Book'))
im = ImageWriter('./temp')


class PDFRender(TextConverter):
    def __init__(self, rsrcmgr):
        super().__init__(rsrcmgr, None)
        self.str_text = ""

    def render(self, item):
        # st.write('container')
        if isinstance(item, LTContainer):
            # st.write('container')
            for child in item:
                self.render(child)
        elif isinstance(item, LTText):
            # st.write('text')
            # st.write(item.get_text())
            self.str_text += item.get_text()
        if isinstance(item, LTTextBox):
            # st.write('textbox')
            # st.write('\n')
            self.str_text += '\n'
        elif isinstance(item, LTImage):
            st.write('image')
            im.export_image(item)

    def receive_layout(self, ltpage):
        self.render(ltpage)


if upload_method == 'File Uploader':
    document_type = st.sidebar.selectbox("Select file format", ('txt', 'pdf'))
    uploaded_file = st.file_uploader('Upload your document here', type=document_type)
    if document_type is 'txt':
        document_file = uploaded_file
    if document_type is 'pdf':
        pdf = PDFParser(uploaded_file)
        document_obj = PDFDocument(pdf)
        rsrcmgr = PDFResourceManager()
        device = PDFRender(rsrcmgr)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(document_obj):
            interpreter.process_page(page)
        st.write(device.str_text)



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
