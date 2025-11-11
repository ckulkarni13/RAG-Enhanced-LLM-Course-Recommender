import os ## This code works
import openai
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Specify your index name
index_name = "course-recommend-proj"

# Check if the index exists, and create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust according to your embedding model's dimensions
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Access the index
index = pc.Index(index_name)

# Streamlit UI
st.title("Query Interface with Knowledge Base")
st.write("This app retrieves responses based on your knowledge base from Pinecone.")

# User input
user_query = st.text_input("Enter your query:")

if user_query:
    st.write("Processing your query...")
    try:
        # Generate the embedding for the user query using OpenAI's model
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # OpenAI model for embeddings
            input=user_query
        )
        query_embedding = response['data'][0]['embedding']

        # Query Pinecone with the embedding
        results = index.query(
            vector=query_embedding,  # The query vector
            top_k=1,  # Return the top 1 most similar result
            include_metadata=True  # Include metadata (if you have it)
        )
        
        # Extract the most relevant document from Pinecone
        if results['matches']:
            best_match = results['matches'][0]
            answer = best_match['metadata']['text']  # Assumes you stored the text as metadata
            st.write("Answer:", answer)
        else:
            st.write("No relevant answers found in the knowledge base.")
        
    except Exception as e:
        st.write(f"Error: {str(e)}")





# import os
# import openai
# import streamlit as st
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set the OpenAI API key from the environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Streamlit UI
# st.title("Simple Query Interface with OpenAI")
# st.write("This app allows you to ask questions, and it responds using OpenAI's GPT model.")

# # User input
# user_query = st.text_input("Enter your query:")

# if user_query:
#     st.write("Processing your query...")
#     try:
#         # Use the new chat API with the correct endpoint
#         response = openai.ChatCompletion.create(
#             model="gpt-4",  # Use GPT-4 if you have access
#             messages=[{"role": "user", "content": user_query}],
#             max_tokens=150
#         )
        
#         # Display the response
#         st.write("Answer:", response['choices'][0]['message']['content'])
        
#     except Exception as e:
#         st.write(f"Error: {str(e)}")




# import streamlit as st
# import openai

# # Set your OpenAI API key
# openai.api_key = 'OPENAI_API_KEY'

# # Streamlit UI
# st.title("Simple Query Interface with OpenAI")
# st.write("This app allows you to ask questions, and it responds using OpenAI's GPT model.")

# # User input
# user_query = st.text_input("Enter your query:")

# if user_query:
#     st.write("Processing your query...")
#     try:
#         # Call OpenAI's GPT model for a response
#         response = openai.Completion.create(
#             model="text-davinci-003",  # Or choose any other model
#             prompt=user_query,
#             max_tokens=150
#         )
        
#         # Display the response
#         st.write("Answer:", response.choices[0].text.strip())
        
#     except Exception as e:
#         st.write(f"Error: {str(e)}")
