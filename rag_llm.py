import streamlit as st
import os
import tempfile
import networkx as nx
import json
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
from langdetect import detect
import re

# Function to extract knowledge graph using LLM
def extract_knowledge_graph_with_llm(documents, ollama_model="llama3.1", language="english"):
    G = nx.Graph()
    entities_added = set()
    
    # Initialize Ollama LLM
    ollama_instance = Ollama(model=ollama_model)
    
    # Create extraction prompt based on language
    if language == "french":
        entity_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Identifiez toutes les entités nommées dans le texte suivant. Les entités peuvent être des personnes, des organisations, des lieux, des dates, etc.
Pour chaque entité, fournissez le type (PERSONNE, ORGANISATION, LIEU, DATE, etc.).

Texte: {text}

Répondez UNIQUEMENT au format JSON exact suivant, sans autre texte avant ou après:
[
  {{"entity": "nom de l'entité", "type": "type de l'entité"}}
]
"""
        )
        
        relationship_extraction_prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template="""
Voici une liste d'entités identifiées dans un texte:
{entities}

Pour le texte suivant, identifiez les relations entre ces entités. Pour chaque relation, indiquez l'entité source, l'entité cible, et une description courte de la relation.

Texte: {text}

Répondez UNIQUEMENT au format JSON exact suivant, sans autre texte avant ou après:
[
  {{"source": "entité source", "target": "entité cible", "relationship": "description de la relation"}}
]
"""
        )
    else:
        entity_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Identify all named entities in the following text. Entities can be people, organizations, locations, dates, etc.
For each entity, provide the type (PERSON, ORGANIZATION, LOCATION, DATE, etc.).

Text: {text}

Respond ONLY with the exact JSON format below, with no text before or after:
[
  {"entity": "entity name", "type": "entity type"}
]
"""
        )
        
        relationship_extraction_prompt = PromptTemplate(
            input_variables=["text", "entities"],
            template="""
Here is a list of entities identified in a text:
{entities}

For the following text, identify relationships between these entities. For each relationship, indicate the source entity, target entity, and a brief description of the relationship.

Text: {text}

Respond ONLY with the exact JSON format below, with no text before or after:
[
  {"source": "source entity", "target": "target entity", "relationship": "relationship description"}
]
"""
        )
    
    # Process each document chunk
    for i, doc in enumerate(documents):
        try:
            with st.spinner(f"Processing chunk {i+1}/{len(documents)}..."):
                content = doc.page_content
                
                # Extract entities
                entity_prompt = entity_extraction_prompt.format(text=content)
                entity_response = ollama_instance.invoke(entity_prompt)
                
                # Parse JSON response - improved error handling
                try:
                    entity_data = safe_json_extract(entity_response)
                    if not entity_data:
                        st.warning(f"Couldn't parse JSON from chunk {i+1} entity response")
                        st.info(f"Raw entity response for chunk {i+1}: {entity_response}")
                        continue

                    # Validate JSON structure
                    if not isinstance(entity_data, list) or not all(isinstance(item, dict) for item in entity_data):
                        st.warning(f"Unexpected JSON structure in chunk {i+1}: {entity_data}")
                        continue
                        
                    # Add entities to graph
                    for item in entity_data:
                        entity = item.get("entity")
                        entity_type = item.get("type")
                        
                        if not entity:
                            st.warning(f"Missing entity in chunk {i+1}: {item}")
                            continue

                        if entity not in entities_added:
                            G.add_node(entity, type=entity_type)
                            entities_added.add(entity)
                    
                    # Extract relationships if we have entities
                    if entity_data:
                        entities_text = ", ".join([item.get("entity") for item in entity_data if item.get("entity")])
                        relationship_prompt = relationship_extraction_prompt.format(
                            text=content,
                            entities=entities_text
                        )
                        relationship_response = ollama_instance.invoke(relationship_prompt)
                        
                        # Parse relationship JSON
                        try:
                            relationship_data = safe_json_extract(relationship_response)
                            if not relationship_data:
                                st.warning(f"Couldn't parse JSON from chunk {i+1} relationship response")
                                continue
                            
                            # Add relationships to graph
                            for item in relationship_data:
                                source = item.get("source")
                                target = item.get("target")
                                relationship = item.get("relationship")
                                
                                if source and target and source in entities_added and target in entities_added and source != target:
                                    if G.has_edge(source, target):
                                        # Increment weight if edge exists
                                        G[source][target]['weight'] += 1
                                        # Append relationship to context
                                        G[source][target]['context'] += f" | {relationship}"
                                    else:
                                        # Create new edge
                                        G.add_edge(source, target, context=relationship, weight=1)
                        except Exception as e:
                            st.warning(f"Error parsing relationships: {str(e)}")
                except Exception as e:
                    st.warning(f"Error parsing entities: {str(e)}")
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
    
    return G

# JSON extraction from LLM responses
def safe_json_extract(text):
    import json
    import re

    # First, try to load the entire text as JSON
    try:
        result = json.loads(text)
        # If it's already a list of dicts with the right structure, return it
        if isinstance(result, list) and all(isinstance(i, dict) for i in result):
            return result
        # If it's a dict with expected fields, wrap it in a list
        elif isinstance(result, dict) and ('entity' in result or 'source' in result):
            return [result]
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON array using regex
    try:
        # Look for array pattern with relaxed whitespace handling
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if array_match:
            return json.loads(array_match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Try to extract individual objects and combine them
    objects = []
    
    # Pattern for entity objects
    entity_matches = re.finditer(r'\{\s*"entity"\s*:\s*"([^"]*)"\s*,\s*"type"\s*:\s*"([^"]*)"\s*\}', text)
    for match in entity_matches:
        objects.append({
            "entity": match.group(1),
            "type": match.group(2)
        })
    
    # Pattern for relationship objects
    rel_matches = re.finditer(r'\{\s*"source"\s*:\s*"([^"]*)"\s*,\s*"target"\s*:\s*"([^"]*)"\s*,\s*"relationship"\s*:\s*"([^"]*)"\s*\}', text)
    for match in rel_matches:
        objects.append({
            "source": match.group(1),
            "target": match.group(2),
            "relationship": match.group(3)
        })
    
    if objects:
        return objects
    
    # If all else fails, return None
    return None

# Function to process document and detect language
def process_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Auto-detect language from the first chunk if possible
    if chunks:
        try:
            sample_text = chunks[0].page_content
            lang_code = detect(sample_text)
            # Map detected language code to our supported languages
            detected_language = "french" if lang_code == "fr" else "english"
            return chunks, detected_language
        except:
            # Default to English if detection fails
            return chunks, "english"
    
    return chunks, "english"

# Function to create vector store from documents
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Function to visualize knowledge graph with pyvis
def visualize_kg(G):
    # Create pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Set node colors based on entity types
    colors = {
        "PERSON": "#ff9999",
        "ORGANIZATION": "#99ff99", 
        "LOCATION": "#9999ff",
        "DATE": "#ffff99",
        "TIME": "#ff99ff",
        "MONEY": "#99ffff",
        "PERCENT": "#ffcc99",
        "Unknown": "#cccccc"
    }
    
    # Add nodes
    for node, attr in G.nodes(data=True):
        entity_type = attr.get('type', 'Unknown').upper()
        net.add_node(
            node, 
            title=node, 
            color=colors.get(entity_type, "#cccccc"),
            size=10 + G.degree(node)
        )
    
    # Add edges with thickness based on weight
    for source, target, attr in G.edges(data=True):
        context = attr.get('context', '')
        weight = attr.get('weight', 1)
        net.add_edge(source, target, title=context[:100] + "...", value=weight)
    
    # Generate visualization in temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name

# Function to generate response from Ollama with RAG
def generate_response(query, vector_store, knowledge_graph, ollama_model="llama3.1", language="english"):
    # Get relevant documents from vector store
    docs = vector_store.similarity_search(query, k=10)
    
    # Get relevant information from knowledge graph
    G = knowledge_graph
    query_terms = query.lower().split()
    relevant_nodes = []
    
    for node in G.nodes():
        node_lower = str(node).lower()
        if any(term in node_lower for term in query_terms):
            relevant_nodes.append(node)
    
    # Extract subgraph of related nodes (up to 2 hops away)
    kg_context = []
    if relevant_nodes:
        for node in relevant_nodes:
            kg_context.append(f"Entity: {node}")
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                context = edge_data.get('context', '')
                kg_context.append(f"Relationship: {node} -- {neighbor}: {context}")
    
    # Combine document context and knowledge graph context
    doc_context = "\n\n".join([doc.page_content for doc in docs])
    kg_context_text = "\n".join(kg_context)
    
    # Create prompt with context based on language
    if language == "french":
        prompt = f"""
Vous êtes un assistant utile qui répond aux questions en fonction des documents fournis.
Répondez à la question suivante en utilisant uniquement les informations du CONTEXTE ci-dessous.
Si vous ne connaissez pas la réponse d'après le contexte, dites "Je n'ai pas assez d'informations pour répondre à cette question."

CONTEXTE DU DOCUMENT:
{doc_context}

CONTEXTE DU GRAPHE DE CONNAISSANCES:
{kg_context_text}

QUESTION: {query}

RÉPONSE:
"""
    else:
        prompt = f"""
You are a helpful assistant that answers questions based on provided documents.
Answer the following question using only the information from the CONTEXT below.
If you don't know the answer based on the context, say "I don't have enough information to answer this question."

DOCUMENT CONTEXT:
{doc_context}

KNOWLEDGE GRAPH CONTEXT:
{kg_context_text}

QUESTION: {query}

ANSWER:
"""
    
    # Call Ollama model
    ollama_instance = Ollama(model=ollama_model)
    response = ollama_instance.invoke(prompt)
    
    return response

# Set page configuration
st.set_page_config(
    page_title="LLM Knowledge Graph RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "show_kg" not in st.session_state:
    st.session_state.show_kg = False
    
if "language" not in st.session_state:
    st.session_state.language = "english"  # Default language

# Sidebar for settings
with st.sidebar:
    st.title("📚 Document Processing")
    
    # Language selection
    language_options = {
        "English": "english",
        "Français": "french"
    }
    selected_language = st.selectbox(
        "Language / Langue",
        options=list(language_options.keys()),
        index=0
    )
    st.session_state.language = language_options[selected_language]
    
    # UI text based on language
    if st.session_state.language == "french":
        upload_text = "Télécharger un document"
        model_text = "Sélectionner un modèle Ollama"
        process_text = "Traiter le document"
        extraction_method_text = "Méthode d'extraction"
    else:
        upload_text = "Upload a document"
        model_text = "Select Ollama Model"
        process_text = "Process Document"
        extraction_method_text = "Extraction Method"
    
    uploaded_file = st.file_uploader(upload_text, type=['pdf', 'txt'])
    
    ollama_model = st.selectbox(
        model_text,
        ["llama3", "llama3.1", "llama3:405b"],
        index=0
    )
    
    # Select extraction method
    extraction_method = st.radio(
        extraction_method_text,
        ["LLM-based", "spaCy-based"],
        index=0
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if st.button(process_text):
            with st.spinner("Processing document..." if st.session_state.language == "english" else "Traitement du document..."):
                try:
                    # Process document and detect language
                    st.session_state.documents, detected_language = process_document(temp_file_path, file_type)
                    
                    # Update language if auto-detected
                    if detected_language != st.session_state.language:
                        # Show message about language detection
                        if st.session_state.language == "english":
                            st.info(f"Detected document language: {detected_language.capitalize()}. Processing with {detected_language.capitalize()} models.")
                        else:
                            st.info(f"Langue détectée: {detected_language.capitalize()}. Traitement avec les modèles {detected_language.capitalize()}.")
                        st.session_state.language = detected_language
                    
                    success_msg = f"Document processed into {len(st.session_state.documents)} chunks" if st.session_state.language == "english" else f"Document traité en {len(st.session_state.documents)} segments"
                    st.success(success_msg)
                    
                    # Create knowledge graph based on selected method
                    with st.spinner("Extracting knowledge graph..." if st.session_state.language == "english" else "Extraction du graphe de connaissances..."):
                        if extraction_method == "LLM-based":
                            st.session_state.knowledge_graph = extract_knowledge_graph_with_llm(
                                st.session_state.documents, 
                                ollama_model=ollama_model, 
                                language=st.session_state.language
                            )
                        else:
                            # Import spaCy for the spaCy-based method
                            import spacy
                            import re
                            
                            # Function to load spaCy model
                            @st.cache_resource
                            def load_nlp_model(language="english"):
                                try:
                                    if language == "english":
                                        return spacy.load("en_core_web_lg")
                                    elif language == "french":
                                        try:
                                            return spacy.load("fr_core_news_lg")
                                        except OSError:
                                            st.info("Downloading French spaCy model. This may take a moment...")
                                            spacy.cli.download("fr_core_news_lg")
                                            return spacy.load("fr_core_news_lg")
                                except OSError:
                                    if language == "english":
                                        st.info("Downloading English spaCy model. This may take a moment...")
                                        spacy.cli.download("en_core_web_lg")
                                        return spacy.load("en_core_web_lg")
                                    elif language == "french":
                                        st.info("Downloading French spaCy model. This may take a moment...")
                                        spacy.cli.download("fr_core_news_lg")
                                        return spacy.load("fr_core_news_lg")
                                return spacy.load("en_core_web_lg")  # Default fallback
                            
                            # Function to extract knowledge graph using spaCy
                            def extract_knowledge_graph_with_spacy(documents, language="english"):
                                nlp = load_nlp_model(language)
                                G = nx.Graph()
                                entities_added = set()
                                
                                for doc in documents:
                                    content = doc.page_content
                                    spacy_doc = nlp(content)
                                    
                                    # Extract named entities
                                    for ent in spacy_doc.ents:
                                        if ent.text.strip() and len(ent.text) > 1:
                                            clean_text = re.sub(r'[^\w\s]', '', ent.text).strip()
                                            if clean_text and clean_text not in entities_added:
                                                G.add_node(clean_text, type=ent.label_)
                                                entities_added.add(clean_text)
                                    
                                    # Extract relationships from sentences
                                    for sent in spacy_doc.sents:
                                        sent_text = sent.text.strip()
                                        if len(sent_text) < 10:  # Skip very short sentences
                                            continue
                                            
                                        ents_in_sent = [ent for ent in spacy_doc.ents 
                                                       if ent.start_char >= sent.start_char 
                                                       and ent.end_char <= sent.end_char]
                                        
                                        # Connect entities that appear in the same sentence
                                        for i, ent1 in enumerate(ents_in_sent):
                                            clean_ent1 = re.sub(r'[^\w\s]', '', ent1.text).strip()
                                            if not clean_ent1 or clean_ent1 not in entities_added:
                                                continue
                                                
                                            for ent2 in ents_in_sent[i+1:]:
                                                clean_ent2 = re.sub(r'[^\w\s]', '', ent2.text).strip()
                                                if not clean_ent2 or clean_ent2 not in entities_added or clean_ent1 == clean_ent2:
                                                    continue
                                                    
                                                # Extract relationship based on the text between entities
                                                if G.has_edge(clean_ent1, clean_ent2):
                                                    # Increment weight if edge exists
                                                    G[clean_ent1][clean_ent2]['weight'] += 1
                                                else:
                                                    # Create new edge with minimal context
                                                    context = sent_text
                                                    G.add_edge(clean_ent1, clean_ent2, context=context, weight=1)
                                
                                return G
                            
                            st.session_state.knowledge_graph = extract_knowledge_graph_with_spacy(
                                st.session_state.documents, 
                                st.session_state.language
                            )
                            
                        num_nodes = len(st.session_state.knowledge_graph.nodes())
                        num_edges = len(st.session_state.knowledge_graph.edges())
                        kg_msg = f"Knowledge graph created with {num_nodes} entities and {num_edges} relationships" if st.session_state.language == "english" else f"Graphe de connaissances créé avec {num_nodes} entités et {num_edges} relations"
                        st.success(kg_msg)
                    
                    # Create vector store
                    with st.spinner("Creating vector database..." if st.session_state.language == "english" else "Création de la base de données vectorielle..."):
                        st.session_state.vector_store = create_vector_store(st.session_state.documents)
                        vdb_msg = "Vector database created" if st.session_state.language == "english" else "Base de données vectorielle créée"
                        st.success(vdb_msg)
                    
                    # Clear chat history
                    st.session_state.messages = []
                    
                except Exception as e:
                    error_msg = f"Error processing document: {str(e)}" if st.session_state.language == "english" else f"Erreur lors du traitement du document: {str(e)}"
                    st.error(error_msg)
                finally:
                    # Clean up temp file
                    os.unlink(temp_file_path)
    
    if st.session_state.knowledge_graph is not None:
        st.divider()
        
        # UI text based on language
        if st.session_state.language == "french":
            kg_title = "📊 Graphe de connaissances"
            show_kg_text = "Afficher le graphe" if not st.session_state.show_kg else "Masquer le graphe"
            entities_text = "Entités"
            relationships_text = "Relations"
            top_entities_text = "Entités principales par connexions:"
            connections_text = "connexions"
        else:
            kg_title = "📊 Knowledge Graph"
            show_kg_text = "Show Knowledge Graph" if not st.session_state.show_kg else "Hide Knowledge Graph"
            entities_text = "Entities"
            relationships_text = "Relationships"
            top_entities_text = "Top entities by connections:"
            connections_text = "connections"
        
        st.subheader(kg_title)
        
        if st.button(show_kg_text):
            st.session_state.show_kg = not st.session_state.show_kg
            
        # Show knowledge graph statistics
        G = st.session_state.knowledge_graph
        st.write(f"{entities_text}: {len(G.nodes())}")
        st.write(f"{relationships_text}: {len(G.edges())}")
        
        # Show top entities by connections
        if len(G.nodes()) > 0:
            st.write(top_entities_text)
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
            for node, degree in top_nodes:
                st.write(f"- {node}: {degree} {connections_text}")

# Main content area
if st.session_state.language == "french":
    main_title = "🧠 Chatbot RAG avec Graphe de Connaissances LLM"
    intro_message = "👈 Veuillez télécharger un document dans la barre latérale pour commencer"
else:
    main_title = "🧠 LLM-based Knowledge Graph RAG Chatbot"
    intro_message = "👈 Please upload a document in the sidebar to get started"

st.title(main_title)

# Display intro message if no documents loaded
if st.session_state.documents == []:
    st.info(intro_message)
    
    if st.session_state.language == "french":
        st.write("""
        ### Comment utiliser cette application:
        1. Téléchargez un document PDF ou TXT dans la barre latérale
        2. Sélectionnez la méthode d'extraction (LLM ou spaCy)
        3. Cliquez sur 'Traiter le document' pour l'analyser
        4. L'application créera un graphe de connaissances et une base de données vectorielle
        5. Vous pourrez ensuite discuter avec votre document à l'aide du formulaire ci-dessous
        6. Les options de visualisation apparaîtront une fois le traitement terminé
        """)
    else:
        st.write("""
        ### How to use this app:
        1. Upload a PDF or TXT document in the sidebar
        2. Select extraction method (LLM or spaCy)
        3. Click 'Process Document' to analyze it
        4. The app will create a knowledge graph and vector database
        5. You can then chat with your document using the form below
        6. Visualization options will appear once processing is complete
        """)
else:
    # Display knowledge graph if enabled
    if st.session_state.show_kg:
        if st.session_state.language == "french":
            st.subheader("Visualisation du Graphe de Connaissances")
            kg_tabs = ["Graphe Interactif", "Table des Entités"]
        else:
            st.subheader("Knowledge Graph Visualization")
            kg_tabs = ["Interactive Graph", "Entity Table"]
        
        # Create tabs for different visualizations
        kg_tab1, kg_tab2 = st.tabs(kg_tabs)
        
        with kg_tab1:
            # Generate interactive graph
            graph_path = visualize_kg(st.session_state.knowledge_graph)
            
            # Display the interactive graph in an iframe
            with open(graph_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            components.html(html_content, height=600)
            
            # Clean up temp file
            os.unlink(graph_path)
            
        with kg_tab2:
            # Create entity table
            G = st.session_state.knowledge_graph
            
            # Create entity dataframe
            entity_data = []
            for node, attr in G.nodes(data=True):
                entity_type = attr.get('type', 'Unknown')
                connections = G.degree(node)
                entity_data.append({
                    "Entity": node,
                    "Type": entity_type,
                    "Connections": connections
                })
            
            entity_df = pd.DataFrame(entity_data)
            if not entity_df.empty:
                st.dataframe(entity_df.sort_values("Connections", ascending=False), use_container_width=True)
            else:
                st.write("No entities found in the knowledge graph.")
    
    # Display chat interface
    st.divider()
    
    if st.session_state.language == "french":
        st.subheader("💬 Discutez avec votre document")
        chat_placeholder = "Posez une question sur votre document..."
        process_doc_message = "Veuillez traiter un document pour activer la fonctionnalité de chat"
        thinking_msg = "Réflexion..."
    else:
        st.subheader("💬 Chat with your document")
        chat_placeholder = "Ask something about your document..."
        process_doc_message = "Please process a document to enable chat functionality"
        thinking_msg = "Thinking..."
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if st.session_state.vector_store is not None:
        user_query = st.chat_input(chat_placeholder)
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(thinking_msg):
                    response = generate_response(
                        user_query, 
                        st.session_state.vector_store, 
                        st.session_state.knowledge_graph,
                        ollama_model,
                        st.session_state.language
                    )
                    st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info(process_doc_message)


# Add footer
st.divider()
footer_text = "LLM-based Knowledge Graph RAG Chatbot | Uses Ollama for both extraction and responses | Built with Streamlit" if st.session_state.language == "english" else "Chatbot RAG avec Graphe de Connaissances LLM | Utilise Ollama pour l'extraction et les réponses | Créé avec Streamlit"
st.caption(footer_text)