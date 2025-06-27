# # from langchain_ollama.llms import OllamaLLM
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# # from vector import retriever

# model = ChatOllama(model="phi3")

# # template = """
# # You are an exeprt in answering questions about a pizza restaurant

# # Here are some relevant reviews: {reviews}

# # Here is the question to answer: {question}
# # """
# # prompt = ChatPromptTemplate.from_template(template)
# # chain = prompt | model

# # while True:
# #     print("\n\n-------------------------------")
# #     question = input("Ask your question (q to quit): ")
# #     print("\n\n")
# #     if question == "q":
# #         break
    
# #     reviews = retriever.invoke(question)
# #     result = chain.invoke({"reviews": reviews, "question": question})
# #     print(result)




# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever  # make sure vector.py is in same folder and defines `retriever`

# # Load the chat model
# model = ChatOllama(model="phi3")  # Ensure "phi3" is downloaded with `ollama pull phi3`

# # Prompt template for question-answering
# template = """
# You are an AI assistant that answers questions **strictly based on the provided employee records** below.
# These records are from a fictional company called TechNova Solutions. The data is entirely synthetic and for testing purposes.

# Your job is to:
# - Use only the information in the provided records.
# - If the answer is not found in the records, reply: "I don't know based on the given data."
# - Do not use any external knowledge or assumptions.

# --- Employee Records ---
# {reviews}
# ------------------------

# Question:
# {question}

# Answer:
# """


# prompt = ChatPromptTemplate.from_template(template)

# # Combine prompt and model into a runnable chain
# chain = prompt | model

# # Chat loop
# while True:
#     print("\n\n-------------------------------")
#     question = input("Ask your question (q to quit): ")
#     print("\n\n")
    
#     if question.strip().lower() == "q":
#         break

#     # Retrieve relevant documents based on question
#     docs = retriever.invoke(question)
#     reviews = "\n\n".join([doc.page_content for doc in docs])

#     # Get model response
#     result = chain.invoke({"reviews": reviews, "question": question})
#     print(result.content)  # `result` is a ChatMessage object; use .content


# chatbot.py























































# main.py

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # <- From vector.py

# Load local model
model = ChatOllama(model="phi3")  # Make sure `ollama pull phi3` is done

# Prompt template
template = """
You are an AI assistant that answers questions strictly based on the employee records below.

These records are from TechNova Solutions and entirely synthetic.

Rules:
- Use only the information in the provided records.
- If the answer is not found, reply: "I don't know based on the given data."
- Do not guess or use external knowledge.

--- Employee Records ---
{reviews}
------------------------

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Build chain
chain = prompt | model

# Q&A Loop
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    print("\n\n")
    
    if question.lower() == "q":
        break

    # Retrieve docs

    # DEBUG: Show what was retrieved
    # Retrieve docs
    docs = retriever.invoke(question)

    # Format retrieved content
    reviews = "\n\n".join([doc.page_content for doc in docs])

    # Get AI response
    result = chain.invoke({"reviews": reviews, "question": question})

    # Only show the final answer
    print("\n[Answer]")
    print(result.content)
