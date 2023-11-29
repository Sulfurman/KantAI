# OpenAI API를 이용한 칸트AI

해당 내용은 OpenAI에서 제공되는 API를 이용해서 철학자 칸트를 재현하는 프로젝트입니다.
(본 프로젝트는 과제 제출용입니다.)

# 참고한 오픈 소스 (앞으로 추가 예정)

- OpenAI Python API library https://github.com/openai/openai-python

- Langchain https://github.com/langchain-ai/langchain

- Langchain Ask PDF https://github.com/alejandro-ao/langchain-ask-pdf
 

# 환경 설정

1. 레포지토리를 클로닝하고, requirements를 설치해 주세요.

```
pip install -r requirements.txt
```

2. OpenAI에서 제공되는 API키를 '.env'파일에 적어주세요.
(해당 API키를 이용하는 건 유료이지만 OpenAI 신규 회원에 한해 5달러 상당의 무료 크레딧이 제공됩니다. 23/11/28 기준)

# 사용법

클로닝한 레포지토리 중 KantBot.py를 실행한 뒤 터미널을 통해 칸트AI에게 질문을 하세요.

# 작동 방식

현재 버전은 크게 3가지로 구성되어 있습니다.

1. pdf를 청크들로 세분화

칸트의 데이터베이스를 학습시키기 위해 미리 준비한 칸트의 서적을 pdf를 텍스트로 추출하고 청크들로 분리해줍니다.

```
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
```

2. 청크들

3. 질문에 대한 대답 생성

유저로부터 얻은 질문을 pdf에서 가장 유사한 내용의 청크들을 호출해 줍니다. 그 다음에 load_qa_chain() 함수를 이용해 얻어낸 청크들로 질문에 대한 대답을 생성합니다.

```
        user_question = st.text_input("질문")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI() #사용할 언어모델
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
```

그 다음 따로 설정한 프롬프트를 앞에서 생성한 대답에 적용시켜 실제 철학자 칸트가 쓸 수 있는 단어나 어투를 설정한뒤 최종적인 대답을 생성, 출력합니다.

```
            my_template = """ You are an AI that reproduces the philosopher Kant.
                              From now on, answer by changing the given content to be the same as Kant,
                              both in tone and level of knowledge.
                              All answers must be in Korean.
                              given content : {sentence} """
                              # AI에게 칸트의 역할을 부여하기 위한 프롬프트 작성

            prompt = PromptTemplate.from_template(my_template)
            prompt.format(sentence = response)
            final = chat_model.predict(prompt.format(sentence = response))

            st.write(final)

```
