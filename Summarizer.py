import streamlit as st
import os
import openai
import tempfile
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader
)
import time

# Streamlit UI
st.set_page_config(
    page_title="Multi-Source AI Summarizer",
    page_icon="📚",
    layout="wide"
)

st.title("🤖 Multi-Source AI Summarizer")
st.markdown("""
This tool can summarize content from various sources including:
- **YouTube Videos** 📹
- **PDF Documents** 📄  
- **CSV Files** 📊
- **Excel Files** 📈
- **Text Files** 📝
- **JSON Files** 📋
- **Web Pages** 🌐
""")

# API Key Section in Sidebar
st.sidebar.header("🔑 API Configuration")

# Groq API Key input
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    help="Get your free API key from https://console.groq.com"
)

# Check if API key is provided
if not groq_api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to continue")
    st.stop()

# Initialize LLM with user-provided API key
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")
    # Test the API key with a simple call
    llm.invoke("Hello")  # Simple test to verify API key
    st.sidebar.success("✅ Groq API key validated!")
except Exception as e:
    st.sidebar.error(f"❌ Invalid Groq API key: {str(e)}")
    st.stop()

# Initialize Hugging Face embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    st.sidebar.success("✅ Hugging Face embeddings loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading embeddings: {str(e)}")
    st.stop()

# Language selection in sidebar
st.sidebar.header("🌍 Language Settings")
summary_language = st.sidebar.selectbox(
    "Select Summary Language:",
    ["English", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese", "Arabic", "Portuguese", "Russian"]
)

# Language mapping for prompt
language_map = {
    "English": "English",
    "Spanish": "Spanish",
    "French": "French", 
    "German": "German",
    "Hindi": "Hindi",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Arabic": "Arabic",
    "Portuguese": "Portuguese",
    "Russian": "Russian"
}

# Dynamic prompt for multilingual summarization
def create_prompt(language):
    return ChatPromptTemplate.from_template(
        f"""
        You are an expert AI assistant specialized in summarizing content from various sources in multiple languages.
        
        Based on the provided context, create a comprehensive and well-structured summary in {language} that captures:
        - Key points and main ideas
        - Important facts and figures
        - Core arguments or conclusions
        - Essential insights
        
        Context:
        {{context}}
        
        Please provide a clear, concise, and informative summary of the content in {language}.
        
        Question/Request: {{input}}
        
        Summary in {language}:
        """
    )

def load_youtube_transcript(video_url):
    """Load transcript from YouTube video - supports multiple languages"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        from urllib.parse import urlparse, parse_qs
        from langchain_core.documents import Document
        
        # Extract video ID
        parsed_url = urlparse(video_url)
        if 'youtube.com' in parsed_url.hostname:
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'youtu.be' in parsed_url.hostname:
            video_id = parsed_url.path[1:]
        else:
            st.error("Invalid YouTube URL")
            return None
        
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None
            
        st.info(f"Extracted Video ID: {video_id}")
        
        try:
            # First, list available transcripts to see languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get manual transcripts first, then generated
            try:
                # Get manual transcript in any language
                transcript = transcript_list.find_manually_created_transcript(['en', 'es', 'fr', 'de', 'hi', 'zh', 'ja', 'ar', 'pt', 'ru'])
                transcript_data = transcript.fetch()
            except:
                # Fallback to auto-generated transcript
                try:
                    transcript = transcript_list.find_generated_transcript(['en', 'es', 'fr', 'de', 'hi', 'zh', 'ja', 'ar', 'pt', 'ru'])
                    transcript_data = transcript.fetch()
                except:
                    # Get any available transcript
                    transcript = list(transcript_list)[0]
                    transcript_data = transcript.fetch()
            
            transcript_text = " ".join([entry["text"] for entry in transcript_data])
            language = transcript.language_code
            
            documents = [Document(
                page_content=transcript_text,
                metadata={
                    "source": video_url, 
                    "title": f"YouTube Video {video_id}",
                    "language": language
                }
            )]
            
            st.success(f"✅ YouTube transcript loaded successfully! (Language: {language})")
            return documents
            
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
            return None
        except NoTranscriptFound:
            st.error("❌ No transcript found for this video.")
            return None
        except VideoUnavailable:
            st.error("❌ The video is unavailable.")
            return None
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return None
                    
    except Exception as e:
        st.error(f"Error loading YouTube transcript: {str(e)}")
        return None

def load_pdf_file(uploaded_file):
    """Load PDF file - supports multiple languages"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Add language metadata
        for doc in documents:
            doc.metadata["language"] = "multilingual"  # PDFs can contain multiple languages
        
        return documents
    except Exception as e:
        st.error(f"Error loading PDF file: {str(e)}")
        return None

def load_csv_file(uploaded_file):
    """Load CSV file - supports multiple languages"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = CSVLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        
        # Add language metadata
        for doc in documents:
            doc.metadata["language"] = "multilingual"
        
        return documents
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def load_excel_file(uploaded_file):
    """Load Excel file - supports multiple languages"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = UnstructuredExcelLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        
        # Add language metadata
        for doc in documents:
            doc.metadata["language"] = "multilingual"
        
        return documents
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def load_text_file(uploaded_file):
    """Load text file - supports multiple languages"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = TextLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        
        # Add language metadata
        for doc in documents:
            doc.metadata["language"] = "multilingual"
        
        return documents
    except Exception as e:
        st.error(f"Error loading text file: {str(e)}")
        return None

def load_json_file(uploaded_file):
    """Load JSON file - supports multiple languages"""
    try:
        # Read and parse JSON file
        json_content = uploaded_file.getvalue().decode('utf-8')
        json_data = json.loads(json_content)
        
        # Convert JSON to readable text
        def json_to_text(data, indent=0):
            text = ""
            if isinstance(data, dict):
                for key, value in data.items():
                    text += "  " * indent + f"{key}: "
                    if isinstance(value, (dict, list)):
                        text += "\n" + json_to_text(value, indent + 1)
                    else:
                        text += f"{value}\n"
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    text += "  " * indent + f"Item {i + 1}:\n"
                    text += json_to_text(item, indent + 1)
            else:
                text += "  " * indent + f"{data}\n"
            return text
        
        json_text = json_to_text(json_data)
        
        from langchain_core.documents import Document
        documents = [Document(
            page_content=json_text,
            metadata={
                "source": uploaded_file.name,
                "language": "multilingual"
            }
        )]
        
        st.success("✅ JSON file loaded successfully!")
        return documents
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def load_web_content(url):
    """Load content from webpage - supports multiple languages"""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Add language metadata
        for doc in documents:
            doc.metadata["language"] = "multilingual"
        
        return documents
    except Exception as e:
        st.error(f"Error loading web content: {str(e)}")
        return None

def create_vector_embedding(documents):
    """Create vector embeddings from documents using Hugging Face"""
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_documents = text_splitter.split_documents(documents)
        
        # Create vector store with Hugging Face embeddings
        vectors = FAISS.from_documents(final_documents, embeddings)
        return vectors
    except Exception as e:
        st.error(f"Error creating vector embeddings: {str(e)}")
        return None

def summarize_content(documents, query, summary_type="comprehensive", language="English"):
    """Generate summary using RAG approach in specified language"""
    try:
        # Create vector embeddings
        vectors = create_vector_embedding(documents)
        if vectors is None:
            return None
        
        # Create prompt with selected language
        prompt = create_prompt(language)
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate summary
        start = time.time()
        response = retrieval_chain.invoke({'input': query})
        processing_time = time.time() - start
        
        st.sidebar.info(f"Processing time: {processing_time:.2f} seconds")
        
        return response
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Sidebar for source selection
st.sidebar.header("📥 Input Source")
source_type = st.sidebar.selectbox(
    "Choose your content source:",
    ["YouTube Video", "PDF File", "CSV File", "Excel File", "Text File", "JSON File", "Web Page"]
)

# Initialize session state for documents
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None

# Source-specific input
documents = None

if source_type == "YouTube Video":
    youtube_url = st.sidebar.text_input("Enter YouTube Video URL:")
    if youtube_url:
        if st.sidebar.button("Load YouTube Transcript"):
            with st.spinner("Loading YouTube transcript..."):
                documents = load_youtube_transcript(youtube_url)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ YouTube transcript loaded successfully! ({len(documents)} documents)")

elif source_type == "PDF File":
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        if st.sidebar.button("Load PDF"):
import streamlit as st
import os
import openai
import tempfile
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader
)
import time

# Streamlit UI
st.set_page_config(
    page_title="Multi-Source AI Summarizer",
    page_icon="📚",
    layout="wide"
)

st.title("🤖 Multi-Source AI Summarizer")
st.markdown("""
This tool can summarize content from various sources including:
- **YouTube Videos** 📹
- **PDF Documents** 📄  
- **CSV Files** 📊
- **Excel Files** 📈
- **Text Files** 📝
- **JSON Files** 📋
- **Web Pages** 🌐
""")

# API Key Section in Sidebar
st.sidebar.header("🔑 API Configuration")

# Groq API Key input
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    help="Get your free API key from https://console.groq.com"
)

# Check if API key is provided
if not groq_api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to continue")
    st.stop()

# Initialize LLM with user-provided API key
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")
    # Test the API key with a simple call
    llm.invoke("Hello")  # Simple test to verify API key
    st.sidebar.success("✅ Groq API key validated!")
except Exception as e:
    st.sidebar.error(f"❌ Invalid Groq API key: {str(e)}")
    st.stop()

# Initialize Hugging Face embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    st.sidebar.success("✅ Hugging Face embeddings loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading embeddings: {str(e)}")
    st.stop()

# NEW: Language selection in sidebar
st.sidebar.header("🌍 Language Settings")
summary_language = st.sidebar.selectbox(
    "Select Summary Language:",
    ["English", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese", "Arabic", "Portuguese", "Russian"]
)

# NEW: Dynamic prompt for multilingual summarization
def create_prompt(language):
    return ChatPromptTemplate.from_template(
        f"""
        You are an expert AI assistant specialized in summarizing content from various sources in multiple languages.
        
        Based on the provided context, create a comprehensive and well-structured summary in {language} that captures:
        - Key points and main ideas
        - Important facts and figures
        - Core arguments or conclusions
        - Essential insights
        
        Context:
        {{context}}
        
        Please provide a clear, concise, and informative summary of the content in {language}.
        
        Question/Request: {{input}}
        
        Summary in {language}:
        """
    )

def load_youtube_transcript(video_url):
    """Load transcript from YouTube video using the correct API methods"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        from urllib.parse import urlparse, parse_qs
        from langchain_core.documents import Document
        
        # Extract video ID
        parsed_url = urlparse(video_url)
        if 'youtube.com' in parsed_url.hostname:
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'youtu.be' in parsed_url.hostname:
            video_id = parsed_url.path[1:]
        else:
            st.error("Invalid YouTube URL")
            return None
        
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None
            
        st.info(f"Extracted Video ID: {video_id}")
        
        try:
            # Use your exact code with proper error handling
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=["en"])
            raw_transcript = fetched.to_raw_data()
            transcript = " ".join(entry["text"] for entry in raw_transcript)
            
            documents = [Document(
                page_content=transcript,
                metadata={"source": video_url, "title": f"YouTube Video {video_id}"}
            )]
            
            st.success("✅ YouTube transcript loaded successfully!")
            return documents
            
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
            return None
        except NoTranscriptFound:
            st.error("❌ No transcript found in the requested language.")
            return None
        except VideoUnavailable:
            st.error("❌ The video is unavailable.")
            return None
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return None
                    
    except Exception as e:
        st.error(f"Error loading YouTube transcript: {str(e)}")
        return None

def load_pdf_file(uploaded_file):
    """Load PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading PDF file: {str(e)}")
        return None

def load_csv_file(uploaded_file):
    """Load CSV file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = CSVLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def load_excel_file(uploaded_file):
    """Load Excel file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = UnstructuredExcelLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def load_text_file(uploaded_file):
    """Load text file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = TextLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading text file: {str(e)}")
        return None

# NEW: JSON file loader function
def load_json_file(uploaded_file):
    """Load JSON file"""
    try:
        # Read and parse JSON file
        json_content = uploaded_file.getvalue().decode('utf-8')
        json_data = json.loads(json_content)
        
        # Convert JSON to readable text
        def json_to_text(data, indent=0):
            text = ""
            if isinstance(data, dict):
                for key, value in data.items():
                    text += "  " * indent + f"{key}: "
                    if isinstance(value, (dict, list)):
                        text += "\n" + json_to_text(value, indent + 1)
                    else:
                        text += f"{value}\n"
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    text += "  " * indent + f"Item {i + 1}:\n"
                    text += json_to_text(item, indent + 1)
            else:
                text += "  " * indent + f"{data}\n"
            return text
        
        json_text = json_to_text(json_data)
        
        from langchain_core.documents import Document
        documents = [Document(
            page_content=json_text,
            metadata={"source": uploaded_file.name}
        )]
        
        st.success("✅ JSON file loaded successfully!")
        return documents
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None

def load_web_content(url):
    """Load content from webpage"""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading web content: {str(e)}")
        return None

def create_vector_embedding(documents):
    """Create vector embeddings from documents using Hugging Face"""
    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_documents = text_splitter.split_documents(documents)
        
        # Create vector store with Hugging Face embeddings
        vectors = FAISS.from_documents(final_documents, embeddings)
        return vectors
    except Exception as e:
        st.error(f"Error creating vector embeddings: {str(e)}")
        return None

def summarize_content(documents, query, summary_type="comprehensive"):
    """Generate summary using RAG approach"""
    try:
        # Create vector embeddings
        vectors = create_vector_embedding(documents)
        if vectors is None:
            return None
        
        # NEW: Use dynamic prompt with selected language
        prompt = create_prompt(summary_language)
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate summary
        start = time.time()
        response = retrieval_chain.invoke({'input': query})
        processing_time = time.time() - start
        
        st.sidebar.info(f"Processing time: {processing_time:.2f} seconds")
        
        return response
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Sidebar for source selection
st.sidebar.header("📥 Input Source")
# NEW: Added "JSON File" to source options
source_type = st.sidebar.selectbox(
    "Choose your content source:",
    ["YouTube Video", "PDF File", "CSV File", "Excel File", "Text File", "JSON File", "Web Page"]
)

# Initialize session state for documents
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None

# Source-specific input
documents = None

if source_type == "YouTube Video":
    youtube_url = st.sidebar.text_input("Enter YouTube Video URL:")
    if youtube_url:
        if st.sidebar.button("Load YouTube Transcript"):
            with st.spinner("Loading YouTube transcript..."):
                documents = load_youtube_transcript(youtube_url)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ YouTube transcript loaded successfully! ({len(documents)} documents)")

elif source_type == "PDF File":
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        if st.sidebar.button("Load PDF"):
            with st.spinner("Loading PDF content..."):
                documents = load_pdf_file(uploaded_pdf)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ PDF content loaded successfully! ({len(documents)} pages)")

elif source_type == "CSV File":
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_csv:
        if st.sidebar.button("Load CSV"):
            with st.spinner("Loading CSV content..."):
                documents = load_csv_file(uploaded_csv)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ CSV content loaded successfully! ({len(documents)} rows)")

elif source_type == "Excel File":
    uploaded_excel = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_excel:
        if st.sidebar.button("Load Excel"):
            with st.spinner("Loading Excel content..."):
                documents = load_excel_file(uploaded_excel)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Excel content loaded successfully! ({len(documents)} sheets)")

elif source_type == "Text File":
    uploaded_text = st.sidebar.file_uploader("Upload Text File", type=["txt"])
    if uploaded_text:
        if st.sidebar.button("Load Text File"):
            with st.spinner("Loading text content..."):
                documents = load_text_file(uploaded_text)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Text content loaded successfully! ({len(documents)} documents)")

# NEW: JSON file upload section
elif source_type == "JSON File":
    uploaded_json = st.sidebar.file_uploader("Upload JSON File", type=["json"])
    if uploaded_json:
        if st.sidebar.button("Load JSON File"):
            with st.spinner("Loading JSON content..."):
                documents = load_json_file(uploaded_json)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ JSON content loaded successfully! ({len(documents)} documents)")

elif source_type == "Web Page":
    web_url = st.sidebar.text_input("Enter Web Page URL:")
    if web_url:
        if st.sidebar.button("Load Web Content"):
            with st.spinner("Loading web content..."):
                documents = load_web_content(web_url)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Web content loaded successfully! ({len(documents)} documents)")

# Summary customization
st.sidebar.header("🎯 Summary Options")
summary_type = st.sidebar.selectbox(
    "Summary Type:",
    ["Comprehensive", "Key Points", "Bullet Points", "Executive Summary"]
)

custom_query = st.sidebar.text_area(
    "Custom Summary Request (optional):",
    placeholder="e.g., Focus on the main arguments and conclusions..."
)

# Main content area
if st.session_state.documents:
    st.header("📝 Generate Summary")
    
    # Default queries based on summary type
    default_queries = {
        "Comprehensive": "Provide a comprehensive summary covering all key aspects of the content.",
        "Key Points": "Extract and summarize the key points and main ideas.",
        "Bullet Points": "Create a bullet-point summary of the most important information.",
        "Executive Summary": "Provide an executive summary highlighting the main findings and conclusions."
    }
    
    query = custom_query if custom_query else default_queries[summary_type]
    
    # NEW: Show selected language in generate button
    if st.button(f"Generate Summary in {summary_language}", type="primary"):
        with st.spinner(f"Generating summary in {summary_language}..."):
            response = summarize_content(st.session_state.documents, query, summary_type)
            
            if response:
                # NEW: Show language in summary header
                st.subheader(f"📋 Summary ({summary_language})")
                st.write(response['answer'])
                
                # Show source documents in expander
                with st.expander("🔍 View Source Content Segments"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Segment {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
else:
    st.info("👆 Please load content using the sidebar options to get started.")

# Instructions
with st.expander("📖 How to use this tool"):
    st.markdown("""
    1. **🔑 Enter your Groq API Key** in the sidebar
    2. **🌍 Select your preferred summary language**
    3. **📥 Select your content source** from the sidebar dropdown
    4. **📄 Provide the content** (upload file, paste URL, etc.)
    5. **🔄 Click the load button** to process your content
    6. **🎯 Choose summary type** or provide custom instructions
    7. **🚀 Click 'Generate Summary'** to get your AI-powered summary
    
    **🌍 NEW: Multilingual Support**
    - Generate summaries in 10 different languages
    - Supports: English, Spanish, French, German, Hindi, Chinese, Japanese, Arabic, Portuguese, Russian
    
    **📋 NEW: JSON File Support**

    
    **📁 Supported Formats:**
    - YouTube videos (automatic transcript extraction)
    - PDF documents
    - CSV files  
    - Excel files (.xlsx, .xls)
    - Text files (.txt)
    - JSON files (.json)
    - Web pages (via URL)
    """)

# Footer
st.markdown("---")
st.markdown(
    "🔒 Your API key is used only for this session and is not stored | " +
    "🌍 Multilingual Support | " +  # NEW: Added multilingual mention
    "Powered by Groq & Llama3 | Built with LangChain & Streamlit",
    unsafe_allow_html=True
)
    - Supports content in any language
    - Generates summaries in 10 different languages
    - Automatic language detection for YouTube videos
    
    **🔧 API Key Information:**
    - **Groq API**: Free tier available at [console.groq.com](https://console.groq.com)
    - **Embeddings**: Free Hugging Face embeddings (no API key required)
    
    **📁 Supported Formats:**
    - YouTube videos (automatic transcript extraction in multiple languages)
    - PDF documents
    - CSV files  
    - Excel files (.xlsx, .xls)
    - Text files (.txt)
    - JSON files (.json)
    - Web pages (via URL)
    """)

# Footer
st.markdown("---")
st.markdown(
    "🔒 Your API key is used only for this session and is not stored | " +
    "🌍 Multilingual Support | " +
    "Powered by Groq & LangChain",
    unsafe_allow_html=True
)
