from dotenv import load_dotenv

from PyPDF2 import PdfReader  #pdf 텍스트 추출
from langchain.text_splitter import CharacterTextSplitter #pdf 파일 청크분류
from langchain.embeddings.openai import OpenAIEmbeddings  #각 청크들을 임베딩
from langchain.vectorstores import FAISS #청크들 중 질문에 가까운 청크 분석
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import streamlit as st
from streamlit_chat import message
import os



def main():
    load_dotenv()

    # UI 헤더 설정
    st.set_page_config(page_title="칸트 AI")
    st.header("칸트 AI")

    #API KEY 설정
    api_key = os.environ.get('OPENAI_API_KEY')
    ChatOpenAI(openai_api_key=api_key)

    #파일 업로드
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    #대답 생성 함수
    def generate_response(user_question):

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

        #대답 생성    
        docs = knowledge_base.similarity_search(user_question)

        llm = ChatOpenAI(model = "gpt-3.5-turbo") #사용할 언어모델
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        my_template = """아래 수칙을 잘 지켜 대답을 생성하시오.
                        - 당신은 철학자 칸트입니다. 칸트의 지식, 업적등을 배경으로 대답한다.
                        - 대답을 할때 아래 주어진 내용을 참고하여 대답한다.
                        - 자신을 소개할때 자신을 AI나 인공지능이라고 소개 하지 않고 진짜 칸트에게 질문한것 처럼 자신을 칸트라고 소개한다.
                        - 너는 칸트이기에 "칸트는 ~" 대신 " 나는 ~"으로 주어를 바꾸어 말한다.
                        - 대답을 할때 가끔씩 "내 생각에는", " ~라고 생각하네" 과 같은 표현을 써서 현실감을 더한다.
                        - 의견을 묻는 질문에 대답을 할때는 칸트가 직접 생각해서 대답한 것처럼 대답하라.
                        - (가장 중요한 수칙) 말할때 " 저는 ~ " 대신 "나는~" 을 사용하고, " ~다 " 로 끝나는대신, "~다네" ," ~라네 " ,"~이라네" , " ~했다네 ", " ~하지 않겠는가 ", " ~ 아니한가 ", " ~하면 좋겠네 " 와 같이 문장을 끝맺음으로써 노인분들이나 스승님같이 진중하고 친근한 반말을 사용한다.
                            주어진 내용 : {sentence}
                        주어진 질문 : {question}"""
            
        chat_prompt = PromptTemplate.from_template(my_template.format(sentence = response, question = user_question))
        chain = LLMChain(
            llm = ChatOpenAI(model = "gpt-3.5-turbo-16k"),
            prompt = chat_prompt
            )            
        final = chain.run(question = user_question, langauge = "Korean")
                            
        return final
    

    # Streamlit UI 생성

    #대화 내용 초기화
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    #사용자 질문 UI
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('질문: ', '', key='input')
        submitted = st.form_submit_button('Send')

    #질문 내용 초기화
    if submitted and user_input:
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    
    #대화 내용 역순으로 표시
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))









if __name__ == '__main__':
    main()
