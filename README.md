# AI-Powered Course Recommender Assistant (RAG Pipeline)

This project implements a state-of-the-art **Retrieval-Augmented Generation (RAG)** pipeline designed to provide hyper-personalized educational recommendations. By leveraging semantic search and large language models, the assistant bridges the gap between natural language user queries and massive course datasets to deliver contextually accurate suggestions.

---

## 🏗 System Architecture & Workflow

The system follows a modular RAG architecture to ensure high-speed retrieval, scalability, and precise response generation.

### 1. Data Ingestion & Semantic Preprocessing
* **Aggregation**: Consolidates course metadata from structured and unstructured sources, including platform-specific datasets and web-scraped content.
* **Cleaning & Normalization**: Utilizes **spaCy** and **NLTK** for advanced text processing, including lemmatization, stop-word removal, and tokenization to prepare raw text for high-fidelity embedding.

### 2. Neural Vectorization & Storage
* **Embedding Model**: Transforms processed course descriptions into 1536-dimensional numerical vectors using the **OpenAI text-embedding-ada-002** model.
* **Vector Database**: Implements **Pinecone** for efficient indexing and storage, enabling sub-second semantic similarity searches through approximate nearest neighbor (ANN) algorithms.

### 3. The RAG Execution Cycle
* **Semantic Retrieval**: User queries are vectorized in real-time and compared against the Pinecone index to pull the top-K most relevant course snippets.
* **Contextual Augmentation**: The retrieved metadata is injected into a specialized "system prompt" to serve as the ground truth for the language model.
* **Intelligent Generation**: An **OpenAI LLM** (GPT-4) processes the augmented prompt, synthesizing the retrieved facts into a natural, persuasive recommendation that explains *why* the courses match the user's specific learning goals.

<p align="center">
  <img src="src/RAG_System_Diagram.png" alt="Course Recommender Project RAG Pipeline Architecture">
</p>

### 4. Interactive Interface
* **Streamlit Frontend**: A responsive web application that allows users to input natural language requests and refine results using dynamic filters (e.g., difficulty level, platform, and duration).

---

## 🚀 Key Technical Features

* **Semantic Intent Recognition**: Moves beyond keyword matching to understand the underlying intent of complex queries, such as "beginner-friendly math for data science."
* **Hallucination Mitigation**: By strictly grounding the LLM's output in retrieved course data, the system ensures recommendations are factual and exist within the database.
* **Metadata Filtering**: Combines vector-based similarity search with traditional metadata filtering to provide a highly granular discovery experience.
* **Scalable Infrastructure**: Designed with a containerized approach using **Docker** for consistent deployment across various environments.

---

## 🛠 Technical Stack

| Component | Technology |
| :--- | :--- |
| **Language Model** | OpenAI GPT-4 / GPT-3.5 |
| **Vector Database** | Pinecone |
| **Embeddings** | OpenAI Ada-002 |
| **Natural Language Processing** | spaCy, NLTK |
| **Data Manipulation** | Pandas, NumPy |
| **Web Framework** | Streamlit |
| **DevOps** | Docker, GitHub Actions |

---

## 📈 Future Enhancements

* **User Profiling**: Integrating historical interaction data to evolve from general RAG to personalized recommendation systems.
* **Cross-Platform API Integration**: Real-time fetching of course availability and pricing from live educational APIs.
* **Hybrid Search**: Combining keyword-based (BM25) and vector-based search to improve retrieval performance for specific technical terms.
