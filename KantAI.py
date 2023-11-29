from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader  #pdf 텍스트 추출
from langchain.text_splitter import CharacterTextSplitter #pdf 파일 청크분류
from langchain.embeddings.openai import OpenAIEmbeddings  #각 청크들을 임베딩
from langchain.vectorstores import FAISS #청크들 중 질문에 가까운 청크 분석
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.chat_models import ChatOpenAI
import os

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    api_key = os.environ.get('OPENAI_API_KEY')

    chat_model = ChatOpenAI(openai_api_key=api_key)
    

    #파일 업로드
    pdf = st.file_uploader("Upload your PDF", type="pdf")


    #텍스트 추출
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    # 청크들로 나누기 
        text_splitter =  CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,    #각 청크가 가지는 글자 개수
            chunk_overlap=200,   #인접한 청크끼리 겹치는 글자 개수 
            length_function=len
        )
        chunks = text_splitter.split_text(text)  #text를 위 splitter로 청크들로 나눔

    #임베딩 생성
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

    #input
        user_question = st.text_input("칸트AI에게 질문하세요")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI() #사용할 언어모델
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            my_template = """ You are an AI that reproduces the philosopher Kant.
                              From now on, answer by changing the given content to be the same as Kant,
                              both in tone and level of knowledge.
                              all answers must be in Korean.
                              given content : {sentence} """
            prompt = PromptTemplate.from_template(my_template)
            prompt.format(sentence = response)
            final = chat_model.predict(prompt.format(sentence = response))

            st.write(final)





if __name__ == '__main__':
    main()



