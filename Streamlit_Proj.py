# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:00:46 2023

@author: HP
"""

import os  # Import os
 
import streamlit as st
from model1 import MyNERModel
import simpletransformers
from simpletransformers.ner import NERModel, NERArgs
import torch


html_temp = """
<h2 style="color:black; text-align:center;">Capstone Project</h2>
<h2 style="color:black;text-align:center;"> Extraction of keywords and summarization of chairman speech of annual reports </h2>
<style>
.stApp {
background-color: white;
color:black;
}
.stButton {
color:white;
}
</style>
"""

st.markdown(html_temp, unsafe_allow_html=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load your NER model using the MyNERModel class
model = MyNERModel("roberta", "roberta-base", ["O", "B-key", "I-key"])
model.load('C:/Users/jadha/Downloads/Capstone/model.pth')  # Replace with the path to your saved model
# model.device = torch.device('cpu')

# Add an upload field to upload a document
uploaded_file = st.file_uploader("Upload a document:", type=["txt", "pdf"])

if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
    st.write("Document Uploaded Succesfully")


    # user_input = st.text_input("Enter a sentence:")
    if st.button("Predict Keywords"):
        keywords = set()
        preds = model.predict([document_text])
        for element in preds[0][0]:
            for key, value in element.items():
                if value[0] == "B" or value[0] == "I":
                    if key[-1].isalnum():
                        keywords.add(key)
                    else:
                        keywords.add(key[:-1])
        
        for keyword in keywords:
            s = ", ".join(keywords)
        
        st.write("The keywords in the text file are: ")
        st.write(s)
        
    if st.button("Summarize"):
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        min_length = int(len(document_text)*0.05)
        max_length = int(len(document_text)*0.1)
        summarized_text = summarizer(document_text, max_length=max_length, min_length=min_length, do_sample=False)
        st.write(summarized_text[0]["summary_text"])