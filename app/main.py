import streamlit as st
from scraper import process_url
from config import Config, logger

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

st.title("IntelliScout - Internet Data Extractor")

# Model selection dropdown with detailed info
st.subheader("Model Selection")
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

url_input = st.text_input("Enter URL to Extract Data From")
prompt = st.text_area("What data do you want to extract?", "Extract information from the given content.")

# Add checkbox for URL processing mode
use_direct_url = st.checkbox(
    "Use Direct URL",
    value=False,
    help="If checked it may not works for large content websites, sends URL directly to model. If unchecked, processes and extracts content first."
)

if st.button("Extract and Analyze"):
    if not url_input or not url_input.strip():
        st.error("Please provide a URL.")
        logger.warning("User submitted empty URL")
        st.stop()

    logger.info(f"User initiated URL extraction for: {url_input} using model: {selected_model_id}")
    st.write(f"Processing: {url_input} with {selected_model['name']}")
    
    with st.spinner("Processing... This may take a moment."):
        result = process_url(url_input, prompt, max_tokens=selected_model['max_tokens'], use_direct_url=use_direct_url)

    st.subheader("Extracted Data")
    st.json(result)

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
