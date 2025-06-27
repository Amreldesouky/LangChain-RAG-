# from langchain_community.embeddings import OllamaEmbeddings
# from chromadb import Client  # or from langchain.vectorstores import Chroma if installed
# from langchain.schema import Document

# import os
# import pandas as pd

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.schema import Document
# import os
# import pandas as pd

# df = pd.read_csv("realistic_restaurant_reviews.csv")

# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chroma_langchain_db"  # fixed typo from chrome_langchain_db
# add_documents = not os.path.exists(db_location)

# if add_documents:
#     documents = []
#     ids = []

#     for i, row in df.iterrows():
#         document = Document(
#             page_content=row["Title"] + " " + row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#             id=str(i)
#         )
#         ids.append(str(i))
#         documents.append(document)

# vector_store = Chroma(
#     collection_name="restaurant_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )

# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)

# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 5}
# )









# import os
# import pandas as pd
# from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# # Load your CSV dataset
# df = pd.read_csv("realistic_restaurant_reviews.csv")

# # Set up embedding model (uses PyTorch only, no TensorFlow needed)
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Set up Chroma DB
# db_location = "./chroma_langchain_db"
# add_documents = not os.path.exists(db_location)

# # Create documents only if database doesn't exist
# if add_documents:
#     documents = []
#     ids = []

#     for i, row in df.iterrows():
#         doc = Document(
#             page_content=row["Title"] + " " + row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#         )
#         documents.append(doc)
#         ids.append(str(i))

# # Create Chroma vector store
# vector_store = Chroma(
#     collection_name="restaurant_reviews",
#     persist_directory=db_location,
#     embedding_function=embedding_model
# )

# # Add documents if needed
# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)

# # Get retriever for later use (RAG)
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# print("✅ Vector DB is ready. Retriever is initialized.")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# import os
# import pandas as pd
# from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# # Load your CSV dataset
# df = pd.read_csv("realistic_restaurant_reviews.csv")

# # Set up embedding model (uses PyTorch only, no TensorFlow needed)
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Set up Chroma DB
# db_location = "./chroma_langchain_db"
# add_documents = not os.path.exists(db_location)

# # Create documents only if database doesn't exist
# if add_documents:
#     documents = []
#     ids = []

#     for i, row in df.iterrows():
#         # Combine relevant fields for text embedding
#         page_text = (
#             f"{row['FirstName']} {row['LastName']} works in {row['Department']} "
#             f"as a {row['Position']} since {row['JoinDate']}. "
#             f"Email: {row['Email']}."
#         )
#         doc = Document(
#             page_content=page_text,
#             metadata={
#                 "department": row["Department"],
#                 "position": row["Position"],
#                 "salary": row["Salary"],
#                 "join_date": row["JoinDate"],
#             },
#         )
#         documents.append(doc)
#         ids.append(str(row["EmployeeID"]))

# # Create Chroma vector store
# vector_store = Chroma(
#     collection_name="employee_profiles",
#     persist_directory=db_location,
#     embedding_function=embedding_model
# )

# # Add documents if needed
# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)

# # Get retriever for later use (RAG)
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# print("✅ Vector DB is ready. Retriever is initialized.")



# # vector.py

























# vector.py

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the correct employee CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")  # <- Make sure this file exists

# Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma DB location
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Build documents from CSV if DB doesn't exist
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Create readable sentence from employee data
        page_text = (
            f"{row['FirstName']} {row['LastName']} works in {row['Department']} "
            f"as a {row['Position']} since {row['JoinDate']}. "
            f"Salary: {row['Salary']}. Email: {row['Email']}."
        )

        doc = Document(
            page_content=page_text,
            metadata={
                "first_name": row["FirstName"],
                "last_name": row["LastName"],
                "department": row["Department"],
                "position": row["Position"],
                "salary": row["Salary"],
                "join_date": row["JoinDate"],
            },
        )
        documents.append(doc)
        ids.append(str(row["EmployeeID"]))

# Create or load the vector store
vector_store = Chroma(
    collection_name="employee_profiles",
    persist_directory=db_location,
    embedding_function=embedding_model
)

# Add documents only if not already in DB
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Use MMR search to improve retrieval
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

print("✅ Vector DB is ready. Retriever is initialized.")
