# LLM Knowledge Graph RAG Chatbot

An advanced document analysis tool that combines Retrieval-Augmented Generation (RAG) with Knowledge Graph extraction for more intelligent document Q&A.


## 🌟 Features

- **Document Processing**: Support for PDF and TXT files
- **Knowledge Graph Extraction**: Two methods for extracting structured information:
  - **LLM-based extraction**: Uses Ollama LLMs to identify entities and relationships
  - **spaCy-based extraction**: Uses spaCy NLP models for entity recognition and relationship extraction
- **Interactive Knowledge Graph Visualization**: Explore relationships between entities with an interactive network graph
- **Multilingual Support**: Works with both English and French documents (auto-detects language)
- **RAG-enhanced Q&A**: Chat with your documents with responses informed by both vector search and knowledge graph context
- **Entity Analytics**: View top entities and their connections in your documents

## 🚀 Why Knowledge Graph RAG?

Traditional RAG systems use vector similarity to retrieve relevant context. This project enhances the retrieval mechanism by adding structured knowledge extraction, allowing the system to:

1. **Understand Relationships**: Identify connections between entities that may not be explicit in the text
2. **Provide Structured Context**: Deliver more precise answers by leveraging entity relationships
3. **Improve Relevance**: Combine vector similarity with graph-based relevance for better context retrieval
4. **Visualize Document Knowledge**: Gain insights into your documents through interactive knowledge graph visualization

## 📋 Requirements

- Python 3.8
- Ollama (with models like llama3, llama3.1, etc.)
- Streamlit
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. Clone this repository:
   

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama following instructions at [ollama.ai](https://ollama.ai)

4. Pull required Ollama models:
   ```bash
   ollama pull llama3
   ollama pull llama3.1 # Optional additional models
   ```

5. Run the application:
   ```bash
   streamlit run llm_rag.py
   ```

## 📖 How to Use

1. **Upload Document**: Select a PDF or TXT file from your local machine
2. **Select Options**:
   - Choose Ollama model (llama3, llama3.1, etc.)
   - Select extraction method (LLM-based or spaCy-based)
   - Select language (or let the app auto-detect)
3. **Process Document**: Click "Process Document" to analyze the text
4. **Explore Knowledge Graph**: View the interactive visualization or entity table
5. **Chat with Document**: Ask questions about your document and receive answers enhanced by both vector search and knowledge graph context

## 🔍 Extraction Methods Compared

### LLM-based Extraction
- **Pros**: More accurate entity and relationship extraction, better understanding of context, works well with complex documents
- **Cons**: Slower processing, requires more computational resources

### spaCy-based Extraction
- **Pros**: Faster processing, works offline, consistent results
- **Cons**: Less accurate for complex relationships, limited to pre-trained entity types

## 🌐 Language Support

The application supports both English and French documents:
- Automatically detects document language
- Uses appropriate prompts and NLP models based on detected language
- Provides UI in the detected language

## 🔧 Technical Details

### Architecture
- **Document Processing**: LangChain for document loading and chunking
- **Vector Storage**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace sentence-transformers
- **Knowledge Graph**: NetworkX for graph representation
- **LLM Integration**: Ollama for local LLM inference
- **Visualization**: Pyvis for interactive graph rendering
- **UI**: Streamlit for web interface

### Performance Considerations
- Processing time depends on document size and extraction method
- LLM-based extraction provides higher quality but is slower
- For large documents, consider using smaller chunk sizes

Created by [Aloys Aymrick NZOOH] - [nzoohaymrick@gmail.com]
