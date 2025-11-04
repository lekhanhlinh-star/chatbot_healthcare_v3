import os
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

# Import shared models
from shared_models import shared_embeddings, shared_chat_model, shared_compressor

# Load vector store with shared embeddings
faiss_path = "faiss_index_document_ckd"
vector_store = FAISS.load_local(faiss_path, shared_embeddings, allow_dangerous_deserialization=True)
print(f"[CKD] Loaded FAISS vector store from: {faiss_path}, id={id(vector_store)}")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

RAG_TEMPLATE = """
##問題(Question):"{question}"

##檢索上下文(Retrieved Context):{context}

##指導方針：
你是一位專精在慢性腎臟病的營養師及藥劑師，能在保持技術準確性的同時，以對話回應問題。
你的任務是當醫生向你提問時，根據檢索上下文中的資訊回答醫生的問題。
檢索上下文當中含有範例問題及回答，你必須對範例回答稍加修飾後輸出。

你必須嚴格遵循以下指導方針：
**你必須優先根據檢索上下文中的資訊回答問題。
**注意回答的篇幅，不需要過長。
**針對問題直接提供問題的解答，避免任何不必要的資訊。
**在使用專有名詞時，必須採用和問題中一樣的說法。
**確保回答內容清晰且精確，並保持專業語氣。
**避免重複，並優先確保內容連貫，清晰地解釋觀點。
**當問題中有出現個別病患的狀況時，必須先總結患者的狀況或該狀況所代表的涵義。
**當沒有明確指示時，你可以假設問題中出現的患者指的是慢性腎臟病患者。
**你說話的對象是醫生，所以你的回應不能要求要與營養師或醫師討論。
**使用繁體中文回答。

根據以上資訊，請回答：{question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Create retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Use shared compressor for contextual compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=shared_compressor, base_retriever=retriever
)

# Create QA chain using shared chat model
qa_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | shared_chat_model
    | StrOutputParser()
)

# related question generate
import re
from langchain_classic.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

related_question_system_prompt = SystemMessagePromptTemplate.from_template(
"""
You are an assistant specializing in question generation, capable of producing multiple relevant and in-depth follow-up questions based on user input. Your goal is to expand the scope of the user's query and cover various aspects of the related field. Please adhere to the following guidelines:

Guidelines:
Question Expansion: Each follow-up question should be closely related to the user's input topic but explore the subject from different angles or levels.
Diversity: Avoid generating overly similar questions. Ensure that the follow-up questions demonstrate diversity in approach and focus.
Specificity: The questions should be specific and actionable, helping the user gain a deeper understanding or exploration of the topic.
Quantity: Generate at least five follow-up questions each time.
Examples:
User Input: "2024 Summer Olympics"
Follow-up Questions:

What new events will be introduced in the 2024 Summer Olympics?
What are the specific dates for the opening and closing ceremonies of the 2024 Olympics?
What changes have been made to the schedule of the 2024 Olympics?
What are the specific events for breakdancing in the 2024 Olympics?
Where will the surfing competitions of the 2024 Olympics take place?
User Input: "How to create charts with Python"
Follow-up Questions:

How can you use Python to create scatter plots?
What other visualization libraries are available in Python?
How do you create line charts in Python?
What are the common issues encountered when creating charts in Python?
How can you add titles and axis labels to charts in Python?
Please follow the above rules to generate relevant follow-up questions based on user input. Do not answer the questions.
Respond in the language specified by the user or the language of the query provided.
"""
)

# Combine system and human prompt templates
human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")
prompt = ChatPromptTemplate.from_messages([related_question_system_prompt, human_prompt])

# Create related question chain using shared chat model
related_question_chain = prompt | shared_chat_model