import streamlit as st
import os
import openai
import tempfile
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
    page_title="OmniSum AI",
    page_icon="📚",  # Changed from ðŸ"š to 📚
    layout="wide"
)

st.title("🤖 Multi-Source Intelligence Platform ")  # Changed from ðŸ¤– to 🤖
st.markdown("""
This tool can summarize content from various sources including:
- **YouTube Videos** 📹
- **PDF Documents** 📄
- **CSV Files** 📊
- **Excel Files** 📈
- **Text Files** 📝
- **Web Pages** 🌐
""")

# API Key Section in Sidebar
st.sidebar.header("🔑 API Configuration")  # Changed from ðŸ"‘ to 🔑

# Groq API Key input
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    help="Get your free API key from https://console.groq.com"
)

# Check if API key is provided
if not groq_api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to continue")  # Changed from ⚠️¸ to ⚠️
    st.stop()

# Initialize LLM with user-provided API key
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")
    # Test the API key with a simple call
    llm.invoke("Hello")  # Simple test to verify API key
    st.sidebar.success("✅ Groq API key validated!")  # Changed from âœ" to ✅
except Exception as e:
    st.sidebar.error(f"❌ Invalid Groq API key: {str(e)}")  # Changed from âŒ to ❌
    st.stop()

# Initialize Hugging Face embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    st.sidebar.success("✅ Hugging Face embeddings loaded!")  # Changed from âœ… to ✅
except Exception as e:
    st.sidebar.error(f"Error loading embeddings: {str(e)}")
    st.stop()

# Updated prompt for summarization
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI assistant specialized in summarizing content from various sources.
    
    Based on the provided context, create a comprehensive and well-structured summary that captures:
    - Key points and main ideas
    - Important facts and figures
    - Core arguments or conclusions
    - Essential insights
    
    Context:
    {context}
    
    Please provide a clear, concise, and informative summary of the content.
    
    Question/Request: {input}
    
    Summary:
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
            fetched = ytt_api.fetch(video_id, languages=["en", "es", "fr", "de", "hi", "zh", "ja", "ar", "pt", "ru"])
            raw_transcript = fetched.to_raw_data()
            transcript = " ".join(entry["text"] for entry in raw_transcript)
            
            documents = [Document(
                page_content=transcript,
                metadata={"source": video_url, "title": f"YouTube Video {video_id}"}
            )]
            
            st.success("✅ YouTube transcript loaded successfully!")  # Changed from âœ… to ✅
            return documents
            
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")  # Changed from âŒ to ❌
            return None
        except NoTranscriptFound:
            st.error("❌ No transcript found in the requested language.")  # Changed from âŒ to ❌
            return None
        except VideoUnavailable:
            st.error("❌ The video is unavailable.")  # Changed from âŒ to ❌
            return None
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")  # Changed from âŒ to ❌
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
st.sidebar.header("📥 Input Source")  # Changed from ðŸ"¥ to 📥
source_type = st.sidebar.selectbox(
    "Choose your content source:",
    ["YouTube Video", "PDF File", "CSV File", "Excel File", "Text File", "Web Page"]
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
                    st.sidebar.success(f"✅ YouTube transcript loaded successfully! ({len(documents)} documents)")  # Changed from âœ… to ✅

elif source_type == "PDF File":
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf:
        if st.sidebar.button("Load PDF"):
            with st.spinner("Loading PDF content..."):
                documents = load_pdf_file(uploaded_pdf)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ PDF content loaded successfully! ({len(documents)} pages)")  # Changed from âœ… to ✅

elif source_type == "CSV File":
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_csv:
        if st.sidebar.button("Load CSV"):
            with st.spinner("Loading CSV content..."):
                documents = load_csv_file(uploaded_csv)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ CSV content loaded successfully! ({len(documents)} rows)")  # Changed from âœ… to ✅

elif source_type == "Excel File":
    uploaded_excel = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_excel:
        if st.sidebar.button("Load Excel"):
            with st.spinner("Loading Excel content..."):
                documents = load_excel_file(uploaded_excel)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Excel content loaded successfully! ({len(documents)} sheets)")  # Changed from âœ… to ✅

elif source_type == "Text File":
    uploaded_text = st.sidebar.file_uploader("Upload Text File", type=["txt"])
    if uploaded_text:
        if st.sidebar.button("Load Text File"):
            with st.spinner("Loading text content..."):
                documents = load_text_file(uploaded_text)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Text content loaded successfully! ({len(documents)} documents)")  # Changed from âœ… to ✅

elif source_type == "Web Page":
    web_url = st.sidebar.text_input("Enter Web Page URL:")
    if web_url:
        if st.sidebar.button("Load Web Content"):
            with st.spinner("Loading web content..."):
                documents = load_web_content(web_url)
                if documents:
                    st.session_state.documents = documents
                    st.sidebar.success(f"✅ Web content loaded successfully! ({len(documents)} documents)")  # Changed from âœ… to ✅

# Summary customization
st.sidebar.header("🎯 Summary Options")  # Changed from ðŸŽ¯ to 🎯
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
    st.header("✍️ Generate Summary")  # Changed from ðŸ" to ✍️
    
    # Default queries based on summary type
    default_queries = {
        "Comprehensive": "Provide a comprehensive summary covering all key aspects of the content.",
        "Key Points": "Extract and summarize the key points and main ideas.",
        "Bullet Points": "Create a bullet-point summary of the most important information.",
        "Executive Summary": "Provide an executive summary highlighting the main findings and conclusions."
    }
    
    query = custom_query if custom_query else default_queries[summary_type]
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            response = summarize_content(st.session_state.documents, query, summary_type)
            
            if response:
                st.subheader("📋 Summary")  # Changed from ðŸ"‹ to 📋
                st.write(response['answer'])
                
                # Show source documents in expander
                with st.expander("🔍 View Source Content Segments"):  # Changed from ðŸ" to 🔍
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Segment {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
else:
    st.info("👆 Please load content using the sidebar options to get started.")  # Changed from ðŸ'† to 👆

# Instructions
with st.expander("📖 How to use this tool"):  # Changed from ðŸ"– to 📖
    st.markdown("""
    1. **🔑 Enter your Groq API Key** in the sidebar  # Changed from ðŸ"' to 🔑
    2. **📥 Select your content source** from the sidebar dropdown  # Changed from ðŸ"¥ to 📥
    3. **📄 Provide the content** (upload file, paste URL, etc.)  # Changed from ðŸ"„ to 📄
    4. **🔄 Click the load button** to process your content  # Changed from ðŸ"„ to 🔄
    5. **🎯 Choose summary type** or provide custom instructions  # Changed from ðŸŽ¯ to 🎯
    6. **🚀 Click 'Generate Summary'** to get your AI-powered summary  # Changed from ðŸš€ to 🚀
    
    **🔧 API Key Information:**  # Changed from ðŸ"§ to 🔧
    - **Groq API**: Free tier available at [console.groq.com](https://console.groq.com)
    - **Embeddings**: Free Hugging Face embeddings (no API key required)
    
    **📋 Supported Formats:**  # Changed from ðŸ" to 📋
    - YouTube videos (automatic transcript extraction)
    - PDF documents
    - CSV files  
    - Excel files (.xlsx, .xls)
    - Text files (.txt)
    - Web pages (via URL)
    """)

# Footer
st.markdown("---")
st.markdown(
    "🔒 Your API key is used only for this session and is not stored | " +  # Changed from ðŸ"' to 🔒
    " Built with LangChain & Streamlit",
    unsafe_allow_html=True
)
