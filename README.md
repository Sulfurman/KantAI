# OpenAI API를 이용한 칸트AI

해당 내용은 OpenAI에서 제공되는 API를 이용해서 철학자 칸트를 재현하는 프로젝트입니다.

(본 프로젝트는 과제 제출용입니다.)

# 참고한 오픈 소스

- OpenAI Python API library https://github.com/openai/openai-python

- Langchain https://github.com/langchain-ai/langchain

- Langchain Ask PDF https://github.com/alejandro-ao/langchain-ask-pdf

- streamlit (UI) https://github.com/streamlit/streamlit
 

# 환경 설정

1. 레포지토리를 클로닝하고, requirements의 내용들을 아래 명령어로 설치해 주세요.

```
pip install -r requirements.txt
```

2. OpenAI에서 제공되는 API키를 '.env'파일에 적어주세요.

(해당 API키를 이용하는 건 유료이지만 OpenAI 신규 회원에 한해 5달러 상당의 무료 크레딧이 제공됩니다. 23/11/28 기준)

3. 자신의 API를 비공개 하려면 gitignore을 삭제하세요.

# 사용법

1. 클로닝한 레포지토리 중 KantBot.py를 실행한 뒤 터미널에 나온 커맨드를 다시 터미널에 작성하세요.  <br/>주의점 : 파일 경로에 띄어쓰기가 있으면 커맨드가 작동하지 않습니다. 띄어 쓰기가 없는 파일 경로로 설정 해주세요.

```
  Warning: to view this Streamlit app on a browser, run it with the following
  command:

    streamlit run (KantAi.py 파일 위치) [ARGUMENTS]
```

2. 칸트AI에게 원하는 pdf파일을(해당 프로젝트에는 칸트의 서적중 하나를 사용했습니다.) 웹페이지에 드래그하여 업로드 하세요.

3. 로딩이 끝나면 칸트AI에게 질문을 하세요.

# 작동 방식

## 현재 버전은 크게 2가지로 구성되어 있습니다.

- UI (Stramlit 라이브러리 사용)
- 대답 생성 함수 (OpenAI 등 라이브러리 사용)

UI를 통해 사용자로부터 질문을 받고 그 질문에 대한 대답 생성 함수에서 생성된 대답을 생성한 뒤 다시 UI를 통해 출력하는 방식입니다.

### 1. UI

1-1. UI의 헤더를 설정

UI 가장 처음 나오는 여러 글자들을 설정합니다.

```
    # UI 헤더 설정
    st.set_page_config(page_title="칸트 AI")
    st.header("칸트 AI")
    subtitle = "칸트가 현대의 지식을 가지고 환생했습니다! \n 그에게 궁금하거나, 필요한 조언을 구해보세요!!"
    st.subheader(subtitle)

    #파일 업로드
    pdf = st.file_uploader("질문하기 전 꼭 text.pdf를 드래그하여 업로드 해주세요!", type="pdf")

```

1-2. 채팅 UI

챗봇과 사용자가 채팅을 할 수 있는 UI를 설정합니다.

```
    #대화 내용 초기화
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    #사용자 질문 UI
    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('질문 ', '', key='input')
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
```


### 2. 대답 생성 함수


2-1. pdf를 청크들로 세분화

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

2-2. 청크들로 임베딩 생성

AI가 청크들을 이용할 수 있게끔 각 청크들을 임베딩합니다.

```
    #임베딩 생성
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
```


2-3. 질문에 대한 대답 생성

유저로부터 얻은 질문을 pdf에서 가장 유사한 내용의 청크들을 호출해 줍니다. 그 다음에 load_qa_chain() 함수를 이용해 얻어낸 청크들로 질문에 대한 대답을 생성합니다.

```
        #대답 생성    
        docs = knowledge_base.similarity_search(user_question)

        llm = ChatOpenAI(model = "gpt-3.5-turbo") #사용할 언어모델
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
```

그 다음 따로 설정한 프롬프트를 앞에서 생성한 대답에 적용시켜 실제 철학자 칸트가 쓸 수 있는 단어나 어투를 설정한뒤 최종적인 대답을 생성, 출력합니다.

```
        my_template = """아래 수칙을 잘 지키고 주어진 내용을 참고하여 대답을 생성하시오.
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
```

# 추가예정 기능

~~1. 디스코드 봇 활용~~

~~현재는 streamlit 을 활용한 웹 안에서 구현되어 있지만 추후에 디스코드 봇에 칸트AI의 답변을 추가 할 예정입니다.~~

(디스코드 방식은 너무 비효율적이어서 streamlit을 그대로 활용해 간단한 채팅 UI를 만들었습니다.)

~~2. 칸트AI의 말투, 지식 개선~~

~~현재 칸트AI에 적용가능한 pdf의 숫자를 늘리고, 말투 또한 더 자연스럽게 개선할 예정입니다.~~

(프롬프트를 더욱 구체적으로 제시하여 말투가 전에 비해 훨씬 자연스러워 졌고, pdf또한 더 정제된 정보를 사용했습니다.)
