import streamlit as st
from scraper import process_url
from config import Config, logger
import os
from PIL import Image
from urllib.parse import urlparse
import re

# Groq models configuration
GROQ_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "name": "LLaMA 4 Scout 17B",
        "max_tokens": 30000,
        "requests_per_minute": 30,
        "tokens_per_minute": 30000
    },
    "allam-2-7b": {
        "name": "Allam 2 7B",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "compound-beta": {
        "name": "Compound Beta",
        "max_tokens": 70000,
        "requests_per_minute": 15,
        "tokens_per_minute": 70000
    },
    "compound-beta-mini": {
        "name": "Compound Beta Mini",
        "max_tokens": 70000,
        "requests_per_minute": 15,
        "tokens_per_minute": 70000
    },
    "deepseek-r1-distill-llama-70b": {
        "name": "Deepseek R1 Distill LLaMA 70B",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "gemma2-9b-it": {
        "name": "Gemma2 9B Instruct",
        "max_tokens": 15000,
        "requests_per_minute": 30,
        "tokens_per_minute": 15000
    },
    "llama-3.1-8b-instant": {
        "name": "LLaMA 3.1 8B Instant",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "llama-3.3-70b-versatile": {
        "name": "LLaMA 3.3 70B Versatile",
        "max_tokens": 12000,
        "requests_per_minute": 30,
        "tokens_per_minute": 12000
    },
    "llama-guard-3-8b": {
        "name": "LLaMA Guard 3 8B",
        "max_tokens": 15000,
        "requests_per_minute": 30,
        "tokens_per_minute": 15000
    },
    "llama3-70b-8192": {
        "name": "LLaMA3 70B (8192)",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "llama3-8b-8192": {
        "name": "LLaMA3 8B (8192)",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "name": "LLaMA 4 Maverick 17B",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },

    "mistral-saba-24b": {
        "name": "Mistral Saba 24B",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    },
    "qwen-qwq-32b": {
        "name": "Qwen QWQ 32B",
        "max_tokens": 6000,
        "requests_per_minute": 30,
        "tokens_per_minute": 6000
    }
}

st.title("IntelliScout - Data Extractor")

# Sidebar for model selection only
with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection dropdown with detailed info
    model_options = {f"{info['name']} (Max: {info['max_tokens']} tokens)": model_id 
                    for model_id, info in GROQ_MODELS.items()}

    selected_model_display = st.selectbox(
        "Select LLM Model",
        options=list(model_options.keys()),
        help="Choose a Groq model to process your data. Each model has different capabilities and limits."
    )

    selected_model_id = model_options[selected_model_display]
    selected_model = GROQ_MODELS[selected_model_id]

    # Display detailed model information
    st.info(f"""
    Model Details:
    - Name: {selected_model['name']}
    - Max Tokens: {selected_model['max_tokens']}
    - Requests/min: {selected_model['requests_per_minute']}
    - Tokens/min: {selected_model['tokens_per_minute']}
    """)

    # Update Config with selected model
    Config.update_model(selected_model_id)

    # Add temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="Controls randomness in the model's output. Lower values make the output more deterministic, higher values make it more creative."
    )
    
    # Update temperature in config
    Config.update_model(selected_model_id, temperature)

# Main content area
st.header("Input Parameters")
url_input = st.text_input("Enter URL to Extract Data From")
prompt = st.text_area("What data do you want to extract?", "Extract information from the given content.")

# Add checkbox for URL processing mode
use_direct_url = st.checkbox(
    "Use Direct URL",
    value=False,
    help="If checked it may not works for large content websites, sends URL directly to model. If unchecked, processes and extracts content first."
)

# Add output format selection
output_format = st.selectbox(
    "Output Format",
    options=["Markdown", "JSON"],
    index=0,  # Default to Markdown
    help="Select the format for the extracted data output"
)

# URL validation
def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check if URL has scheme and netloc
        if not all([result.scheme, result.netloc]):
            return False
            
        # Check if domain has proper format (e.g., example.com)
        domain_parts = result.netloc.split('.')
        if len(domain_parts) < 2:
            return False
            
        # Check if domain parts are valid
        for part in domain_parts:
            if not re.match(r'^[a-zA-Z0-9-]+$', part):
                return False
            if part.startswith('-') or part.endswith('-'):
                return False
                
        # Check if TLD is at least 2 characters
        if len(domain_parts[-1]) < 2:
            return False
            
        return True
    except:
        return False

# Validate URL and control button state
is_url_valid = is_valid_url(url_input) if url_input else False
button_disabled = not is_url_valid

# Process button in main content
process_button = st.button(
    "Extract and Analyze",
    type="primary",
    disabled=button_disabled,
    help="Enter a valid URL to enable extraction" if button_disabled else "Click to extract and analyze"
)

# Show validation message if URL is invalid
if url_input and not is_url_valid:
    st.error("Please enter a valid URL (e.g., https://example.com). The URL must have a proper domain format with at least one dot (.) and valid characters.")

if process_button:
    if not url_input or not url_input.strip():
        st.error("Please provide a URL.")
        logger.warning("User submitted empty URL")
        st.stop()

    logger.info(f"User initiated URL extraction for: {url_input} using model: {selected_model_id}")
    st.write(f"Processing: {url_input} with {selected_model['name']}")
    
    with st.spinner("Processing... This may take a moment."):
        result = process_url(url_input, prompt, max_tokens=selected_model['max_tokens'], use_direct_url=use_direct_url, output_format=output_format.lower())

    # Display screenshot if available
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.subheader("Extracted Data")
        if output_format.lower() == "json":
            st.json(result)
        else:  # Markdown format
            st.markdown(result)

        # Display token usage if available
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            input_tokens = usage.get("prompt_tokens", "N/A")
            output_tokens = usage.get("completion_tokens", "N/A")
            total_tokens = usage.get("total_tokens", "N/A")
            
            st.subheader("Token Usage")
            st.write(f"Input Tokens: {input_tokens}")
            st.write(f"Output Tokens: {output_tokens}")
            st.write(f"Total Tokens: {total_tokens}")

            # Warn if approaching token limit
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                used_tokens = input_tokens + output_tokens
                if used_tokens > selected_model['max_tokens'] * 0.9:  # Warn at 90% of limit
                    st.warning(f"Warning: Used {used_tokens} tokens, nearing limit of {selected_model['max_tokens']}")

    with col2:
        st.subheader("Page Preview")
        # Check if extraction was successful
        if isinstance(result, dict) and result.get("status") == "success":
            screenshot_path = "screenshots/screenshot.png"
            if os.path.exists(screenshot_path):
                # Set smaller fixed dimensions
                fixed_height = 200
                fixed_width = 200
                
                # Display the image with fixed dimensions
                clicked = st.image(screenshot_path, 
                        caption="Click to expand",
                        width=fixed_width,
                        use_container_width=False)
                
            else:
                st.error("Screenshot not available")
        else:
            # For markdown output, we'll still show the screenshot if it exists
            screenshot_path = "screenshots/screenshot.png"
            if os.path.exists(screenshot_path):
                # Set smaller fixed dimensions
                fixed_height = 200
                fixed_width = 200
                
                # Display the image with fixed dimensions
                clicked = st.image(screenshot_path, 
                        caption="Click to expand",
                        width=fixed_width,
                        use_container_width=False)
            else:
                st.error("Screenshot not available")
            