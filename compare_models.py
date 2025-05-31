import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time

# Initialize the Groq client
client = Groq(api_key='')

# Model information dictionary
MODEL_INFO = {
    "gemma2-9b-it": {
        "name": "Gemma 2 9B Instruct",
        "type": "Text Generation",
        "parameters": "9 billion",
        "description": "Google's instruction-tuned model optimized for conversational AI and text generation tasks.",
        "best_for": "General conversation, creative writing, code generation",
        "strengths": "Fast inference, good at following instructions, multilingual support"
    },
    "meta-llama/Llama-Guard-4-12B": {
        "name": "Llama Guard 4 12B",
        "type": "Safety & Moderation",
        "parameters": "12 billion",
        "description": "Meta's safety-focused model designed for content moderation and harmful content detection.",
        "best_for": "Content safety, moderation, policy compliance checking",
        "strengths": "Excellent safety detection, policy-aware responses"
    },
    "llama-3.3-70b-versatile": {
        "name": "Llama 3.3 70B Versatile",
        "type": "Text Generation",
        "parameters": "70 billion",
        "description": "Meta's large-scale model with versatile capabilities across multiple domains and tasks.",
        "best_for": "Complex reasoning, research, detailed analysis, professional writing",
        "strengths": "High-quality outputs, strong reasoning, comprehensive knowledge"
    },
    "llama-3.1-8b-instant": {
        "name": "Llama 3.1 8B Instant",
        "type": "Text Generation",
        "parameters": "8 billion",
        "description": "Optimized for speed while maintaining quality, ideal for real-time applications.",
        "best_for": "Quick responses, chatbots, real-time applications",
        "strengths": "Ultra-fast inference, good balance of speed and quality"
    },
    "llama3-70b-8192": {
        "name": "Llama 3 70B",
        "type": "Text Generation",
        "parameters": "70 billion",
        "description": "Meta's flagship model with extended context window for handling long documents.",
        "best_for": "Long document analysis, detailed summarization, complex tasks",
        "strengths": "Large context window (8192 tokens), excellent for long texts"
    },
    "llama3-8b-8192": {
        "name": "Llama 3 8B",
        "type": "Text Generation",
        "parameters": "8 billion",
        "description": "Efficient model with good performance-to-speed ratio and extended context.",
        "best_for": "Balanced tasks, medium-length documents, efficient processing",
        "strengths": "Good balance of speed and capability, extended context support"
    },
    "whisper-large-v3": {
        "name": "Whisper Large v3",
        "type": "Speech-to-Text",
        "parameters": "1.5 billion",
        "description": "OpenAI's advanced speech recognition model with high accuracy across languages.",
        "best_for": "Audio transcription, multilingual speech recognition",
        "strengths": "High accuracy, 99+ languages, robust to noise and accents"
    },
    "whisper-large-v3-turbo": {
        "name": "Whisper Large v3 Turbo",
        "type": "Speech-to-Text",
        "parameters": "1.5 billion",
        "description": "Optimized version of Whisper Large v3 for faster transcription with maintained accuracy.",
        "best_for": "Fast audio transcription, real-time applications",
        "strengths": "8x faster than standard Whisper, maintains high accuracy"
    },
    "distil-whisper-large-v3-en": {
        "name": "Distil-Whisper Large v3 English",
        "type": "Speech-to-Text",
        "parameters": "756 million",
        "description": "Distilled version of Whisper optimized specifically for English transcription.",
        "best_for": "English-only transcription, resource-efficient applications",
        "strengths": "Faster than full Whisper, optimized for English, lower resource usage"
    }
}


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_url(url):
    """
    Extract text content from a website URL
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = "https://" + url

        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        # Extract text from common content containers
        content_selectors = [
            'main', 'article', '.content', '#content',
            '.post-content', '.entry-content', '.article-content',
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        ]

        extracted_text = ""

        # Try to find main content first
        for selector in ['main', 'article', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                extracted_text = main_content.get_text(separator=' ', strip=True)
                break

        # If no main content found, get all paragraph and heading text
        if not extracted_text:
            elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            extracted_text = ' '.join([elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)])

        # Clean up the text
        lines = extracted_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                cleaned_lines.append(line)

        final_text = '\n'.join(cleaned_lines)

        if not final_text or len(final_text) < 100:
            raise Exception("Could not extract sufficient text content from the webpage")

        return final_text

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch webpage: {str(e)}")
    except Exception as e:
        raise Exception(f"Error extracting text from URL: {str(e)}")


def validate_url(url):
    """
    Validate if the provided URL is properly formatted
    """
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme in ['http', 'https'])
    except:
        return False


def summarize_text(text, model_name):
    try:
        summary_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Provide a concise and comprehensive summary."
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text: {text}"
                }
            ],
            model=model_name,
        )
        return summary_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def ask_question(context, question, model_name):
    try:
        answer_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based on the provided context accurately and comprehensively."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}"
                }
            ],
            model=model_name,
        )
        return answer_response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def display_model_card(model_key, model_info):
    """Display a model information card"""
    with st.container():
        st.markdown(f"""
        <div style="
            border: 1px solid #e0e0e0; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 10px 0; 
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">{model_info['name']}</h4>
            <p style="margin: 5px 0;"><strong>Type:</strong> <span style="color: #3498db;">{model_info['type']}</span></p>
            <p style="margin: 5px 0;"><strong>Parameters:</strong> <span style="color: #e74c3c;">{model_info['parameters']}</span></p>
            <p style="margin: 10px 0;"><strong>Description:</strong> {model_info['description']}</p>
            <p style="margin: 5px 0;"><strong>Best for:</strong> <em>{model_info['best_for']}</em></p>
            <p style="margin: 5px 0;"><strong>Key Strengths:</strong> <em>{model_info['strengths']}</em></p>
        </div>
        """, unsafe_allow_html=True)


def get_recommended_models(task_type):
    """Get recommended models for specific tasks"""
    recommendations = {
        "summarization": ["llama-3.3-70b-versatile", "llama3-70b-8192", "gemma2-9b-it"],
        "question_answering": ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"]
    }
    return recommendations.get(task_type, [])


# Streamlit UI Configuration
st.set_page_config(
    page_title="AI PDF Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered PDF Analyzer</h1>
    <p>Extract insights from your documents using advanced AI models</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.header("🔧 Navigation")
    page = st.radio("Choose a section:", ["📄 Content Analysis", "🧠 Know Your Models", "ℹ️ About"])

if page == "🧠 Know Your Models":
    st.markdown('<div class="section-header"><h2>🧠 Know Your Models</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Understanding the capabilities of different AI models helps you choose the right tool for your task. 
    Each model has unique strengths and is optimized for specific use cases.
    """)

    # Filter models by type
    model_types = list(set([info['type'] for info in MODEL_INFO.values()]))
    selected_type = st.selectbox("Filter by model type:", ["All"] + model_types)

    # Display models
    for model_key, model_info in MODEL_INFO.items():
        if selected_type == "All" or model_info['type'] == selected_type:
            display_model_card(model_key, model_info)

    # Model comparison section
    st.markdown('<div class="section-header"><h3>📊 Quick Model Comparison</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🚀 For Speed:**")
        st.markdown("• llama-3.1-8b-instant\n• gemma2-9b-it\n• whisper-large-v3-turbo")

        st.markdown("**🎯 For Accuracy:**")
        st.markdown("• llama-3.3-70b-versatile\n• llama3-70b-8192\n• whisper-large-v3")

    with col2:
        st.markdown("**📝 For Long Documents:**")
        st.markdown("• llama3-70b-8192\n• llama3-8b-8192")

        st.markdown("**🛡️ For Safety:**")
        st.markdown("• meta-llama/Llama-Guard-4-12B")

elif page == "ℹ️ About":
    st.markdown('<div class="section-header"><h2>ℹ️ About This Application</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Purpose
    This application leverages cutting-edge AI models to help you extract valuable insights from PDF documents through:
    - **Intelligent Summarization**: Get concise summaries of lengthy documents
    - **Interactive Q&A**: Ask specific questions about your document content

    ### 🔧 Technology Stack
    - **Frontend**: Streamlit for the user interface
    - **PDF Processing**: PyMuPDF for text extraction
    - **AI Models**: Groq API for fast inference
    - **Multiple Models**: Choose from 9 different AI models for various tasks

    ### 💡 Tips for Best Results
    - Choose larger models (70B parameters) for complex documents
    - Use faster models (8B parameters) for quick summaries
    - Whisper models are designed for audio transcription, not text analysis
    - For safety-critical content, consider using Llama-Guard models

    ### 🚀 Getting Started
    1. Upload your PDF document
    2. Choose appropriate models for your tasks
    3. Generate summaries or ask questions
    4. Explore different models to find what works best for your needs
    """)

else:  # Content Analysis page
    st.markdown('<div class="section-header"><h2>📄 Content Analysis</h2></div>', unsafe_allow_html=True)

    # Display image if available
    try:
        image = Image.open('image.png')
        st.image(image, use_container_width=True)
    except:
        pass

    # Source selection
    st.markdown("### 📋 Choose Your Content Source")
    source_type = st.selectbox(
        "What would you like to analyze?",
        ["📄 PDF Document", "🌐 Website URL"],
        help="Select whether you want to analyze a PDF file or extract content from a website"
    )

    extracted_text = ""
    source_info = {}

    if source_type == "📄 PDF Document":
        uploaded_file = st.file_uploader("📁 Upload a PDF file", type="pdf", help="Select a PDF document to analyze")

        if uploaded_file is not None:
            # Extract text from PDF
            with st.spinner("🔄 Extracting text from PDF..."):
                try:
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    source_info = {
                        "type": "PDF",
                        "name": uploaded_file.name,
                        "size": len(uploaded_file.getvalue())
                    }
                    st.success("✅ PDF text extracted successfully!")
                except Exception as e:
                    st.error(f"❌ Error extracting PDF text: {str(e)}")

    elif source_type == "🌐 Website URL":
        st.markdown("### 🌐 Website Analysis")

        url_input = st.text_input(
            "Enter website URL:",
            placeholder="https://example.com or example.com",
            help="Enter the URL of the website you want to analyze. HTTP/HTTPS prefix is optional."
        )

        if url_input:
            # Add protocol if missing
            if not url_input.startswith(('http://', 'https://')):
                formatted_url = f"https://{url_input}"
            else:
                formatted_url = url_input

            if st.button("🔄 Extract Website Content", type="primary"):
                with st.spinner("🔄 Extracting content from website..."):
                    try:
                        extracted_text = extract_text_from_url(formatted_url)
                        source_info = {
                            "type": "URL",
                            "name": formatted_url,
                            "domain": urlparse(formatted_url).netloc
                        }
                        st.success(f"✅ Successfully extracted content from {source_info['domain']}")
                    except Exception as e:
                        st.error(f"❌ Error extracting website content: {str(e)}")
                        st.markdown("""
                        **Troubleshooting Tips:**
                        - Make sure the URL is correct and accessible
                        - Some websites block automated access
                        - Try a different URL or check if the site is down
                        - Ensure the website contains readable text content
                        """)

    # Display content information and analysis options
    if extracted_text:
        # Display content info
        col1, col2, col3 = st.columns(3)
        with col1:
            if source_info["type"] == "PDF":
                st.metric("📄 Source", "PDF Document")
                st.caption(source_info["name"])
            else:
                st.metric("🌐 Source", "Website")
                st.caption(source_info["domain"])

        with col2:
            st.metric("📝 Characters", f"{len(extracted_text):,}")

        with col3:
            st.metric("📊 Words (approx)", f"{len(extracted_text.split()):,}")

        # Text preview
        with st.expander("👀 Preview extracted content"):
            preview_text = extracted_text[:1500] + "..." if len(extracted_text) > 1500 else extracted_text
            st.text_area("Content preview:", preview_text, height=200, disabled=True)

        # Summarization Section
        st.markdown('<div class="section-header"><h3>📝 Content Summarization</h3></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            summarize_model = st.selectbox(
                "Choose a model for summarization:",
                ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192",
                 "meta-llama/Llama-Guard-4-12B"],
                help="Select the AI model that best fits your summarization needs"
            )

        with col2:
            # Show model recommendations
            recommended = get_recommended_models("summarization")
            if summarize_model in recommended:
                st.markdown('<div class="recommendation-box">✅ Recommended for summarization</div>',
                            unsafe_allow_html=True)

        # Display selected model info
        if summarize_model in MODEL_INFO:
            st.info(f"**{MODEL_INFO[summarize_model]['name']}**: {MODEL_INFO[summarize_model]['description']}")

        # Summarization options
        col1, col2 = st.columns(2)
        with col1:
            summary_length = st.selectbox(
                "Summary length:",
                ["Brief", "Detailed", "Comprehensive"],
                help="Choose how detailed you want the summary to be"
            )

        with col2:
            summary_focus = st.selectbox(
                "Summary focus:",
                ["General", "Key Points", "Main Arguments", "Statistics & Data", "Conclusions"],
                help="Choose what aspect to focus on in the summary"
            )

        summary_button = st.button("🔄 Generate Summary", type="primary")

        if summary_button:
            with st.spinner("🤖 Generating summary..."):
                # Create enhanced prompt based on user preferences
                length_instructions = {
                    "Brief": "Provide a brief, concise summary in 2-3 sentences.",
                    "Detailed": "Provide a detailed summary covering all major points.",
                    "Comprehensive": "Provide a comprehensive summary with thorough analysis and context."
                }

                focus_instructions = {
                    "General": "Focus on the overall content and main themes.",
                    "Key Points": "Focus on identifying and summarizing the key points and main ideas.",
                    "Main Arguments": "Focus on the main arguments, claims, and reasoning presented.",
                    "Statistics & Data": "Focus on numerical data, statistics, and quantitative information.",
                    "Conclusions": "Focus on conclusions, recommendations, and final outcomes."
                }

                enhanced_prompt = f"""
                Please summarize the following content with these specifications:
                - Length: {length_instructions[summary_length]}
                - Focus: {focus_instructions[summary_focus]}

                Content: {extracted_text}
                """

                try:
                    # For very long content, truncate to avoid token limits
                    max_chars = 15000  # Adjust based on model's token limit
                    content_to_summarize = extracted_text[:max_chars] if len(
                        extracted_text) > max_chars else extracted_text

                    final_prompt = f"""
                    {length_instructions[summary_length]}
                    {focus_instructions[summary_focus]}

                    Please summarize the following content:

                    {content_to_summarize}
                    """

                    summary_response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant skilled at creating clear, informative summaries. Follow the user's specifications for length and focus."
                            },
                            {
                                "role": "user",
                                "content": final_prompt
                            }
                        ],
                        model=summarize_model,
                    )
                    summary = summary_response.choices[0].message.content

                    # Show source information with summary
                    st.markdown("### 📋 Summary:")
                    if source_info["type"] == "URL":
                        st.caption(f"📄 Summary of content from: {source_info['domain']}")
                    else:
                        st.caption(f"📄 Summary of: {source_info['name']}")

                    st.write(summary)

                    # Option to save summary
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save Summary"):
                            filename = f"summary_{source_info.get('domain', 'content')}.txt" if source_info[
                                                                                                    'type'] == 'URL' else f"summary_{source_info['name'][:20]}.txt"
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=filename,
                                mime="text/plain"
                            )
                    with col2:
                        st.metric("📊 Summary Length", f"{len(summary.split())} words")

                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")

        # Question Answering Section
        st.markdown('<div class="section-header"><h3>❓ Question & Answer</h3></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            question_model = st.selectbox(
                "Choose a model for question answering:",
                ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192",
                 "meta-llama/Llama-Guard-4-12B"],
                help="Select the AI model for answering your questions"
            )

        with col2:
            # Show model recommendations
            recommended = get_recommended_models("question_answering")
            if question_model in recommended:
                st.markdown('<div class="recommendation-box">✅ Recommended for Q&A</div>', unsafe_allow_html=True)

        # Display selected model info
        if question_model in MODEL_INFO:
            st.info(f"**{MODEL_INFO[question_model]['name']}**: {MODEL_INFO[question_model]['description']}")

        question = st.text_input("💬 Ask a question about the content:",
                                 placeholder="What is the main topic of this content?")

        if question:
            with st.spinner("🔍 Finding answer..."):
                try:
                    answer = ask_question(extracted_text, question, question_model)
                    st.markdown("### 💡 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

    else:
        # Show instructions and examples
        if source_type == "📄 PDF Document":
            st.info("👆 Please upload a PDF file to begin analysis.")
        else:
            st.info("👆 Please enter a website URL to begin analysis.")

            # Show example URLs
            st.markdown("### 🌐 Example URLs you can try:")
            example_urls = [
                "wikipedia.org/wiki/Artificial_intelligence",
                "news.ycombinator.com",
                "medium.com/@username/article-title",
                "blog.example.com/post-title",
                "docs.example.com/documentation"
            ]

            for url in example_urls:
                st.markdown(f"• {url}")

        # Show sample questions while waiting
        st.markdown("### 💡 Sample Questions You Can Ask:")
        sample_questions = [
            "What are the key points discussed in this content?",
            "Who are the main people or organizations mentioned?",
            "What conclusions or recommendations are made?",
            "What are the main statistics or data points?",
            "What is the content's purpose or objective?",
            "Summarize the main arguments presented"
        ]

        for q in sample_questions:
            st.markdown(f"• {q}")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: Different models excel at different tasks. Experiment with various models to find the best fit for your specific needs!")
