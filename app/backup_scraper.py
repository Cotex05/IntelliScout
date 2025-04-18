import os
import time
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright
from htmlmin import minify
from markdownify import markdownify as md
from scrapegraphai.graphs import SmartScraperGraph
from config import Config, logger
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def save_screenshot(page, url):
    """Save a screenshot of the opened page to a directory."""
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_filename = os.path.join(screenshot_dir, f"screenshot.png")
    try:
        page.screenshot(path=screenshot_filename, full_page=True)
        logger.info(f"Saved screenshot to {screenshot_filename}")
    except Exception as e:
        logger.error(f"Error saving screenshot for {url}: {str(e)}")


def extract_html_with_playwright(url, max_retries=3, retry_delay=5):
    """Extract raw HTML from a URL using Playwright with enhanced SPA support.

    Args:
        url (str): The URL to scrape
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    """
    logger.info(f"Extracting HTML from: {url}")

    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-gpu',
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process'
                    ]
                )
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    java_script_enabled=True,
                )

                # Add extra headers to appear more like a real browser
                context.set_extra_http_headers({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache',
                })

                page = context.new_page()

                # Set longer timeout for modern SPAs
                page.set_default_timeout(45000)  # 45 seconds
                page.set_default_navigation_timeout(45000)

                # Navigate to the page
                response = page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=45000
                )

                if not response or not response.ok:
                    raise Exception(
                        f"Failed to get response from {url}. Status: {response.status if response else 'No response'}")

                # Wait for key elements to be present
                try:
                    # Wait for body to be loaded
                    page.wait_for_selector('body', timeout=10000)

                    # Additional waits for SPAs
                    page.wait_for_load_state('domcontentloaded')
                    page.wait_for_load_state('networkidle')

                    # Wait a bit more for dynamic content
                    page.wait_for_timeout(2000)

                    # Scroll to bottom to trigger lazy loading
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)

                except Exception as e:
                    logger.warning(f"Some wait conditions failed: {str(e)}, but continuing...")

                # Take screenshot for debugging
                save_screenshot(page, url)

                # Get the page content
                html_content = page.content()

                # Verify we have meaningful content
                if not html_content or len(html_content.strip()) < 100:
                    raise ValueError("Received empty or too short HTML content")

                # Check if we have a body tag
                if "<body" not in html_content.lower():
                    raise ValueError("No HTML body tag found in the content")

                # Parse with BeautifulSoup to verify content
                soup = BeautifulSoup(html_content, "html.parser")
                body = soup.find('body')

                if not body:
                    raise ValueError("No HTML body content found after parsing")

                if not body.get_text(strip=True):
                    raise ValueError("HTML body contains no text content")

                # Clean up
                context.close()
                browser.close()

                logger.info(f"Successfully extracted HTML from {url}. Length: {len(html_content)} chars")
                logger.info(f"Content preview: {html_content[:500]}...")
                return html_content

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All attempts failed for {url}")
                raise Exception(f"Failed to extract HTML after {max_retries} attempts: {str(e)}")


def clean_html(html_content):
    """Clean up HTML content by removing unwanted elements while preserving important content.

    Args:
        html_content (str): Raw HTML content to clean
    Returns:
        str: Cleaned HTML content
    """
    logger.info(f"Cleaning HTML content. Original length: {len(html_content)} chars")
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted tags
        unwanted_tags = [
            "script", "style", "iframe", "noscript", "meta", "link", "footer",
            "header", "nav", "aside", "advertisement", "banner"
        ]
        for tag in soup.find_all(unwanted_tags):
            tag.decompose()

        # Remove all HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove empty tags except those that might be meaningful
        meaningful_empty_tags = ["br", "hr", "img", "input"]
        for tag in soup.find_all():
            if not tag.get_text(strip=True) and tag.name not in meaningful_empty_tags:
                if not tag.find_all(meaningful_empty_tags):
                    tag.decompose()

        # Remove hidden elements
        for tag in soup.find_all(style=True):
            if 'display: none' in tag.get('style', '') or 'visibility: hidden' in tag.get('style', ''):
                tag.decompose()

        # Remove elements with specific class names often used for unwanted content
        unwanted_classes = [
            'ad', 'ads', 'advertisement', 'banner', 'cookie', 'popup', 'modal',
            'social-share', 'newsletter', 'sidebar'
        ]
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=lambda x: x and class_name in x.lower()):
                element.decompose()

        # Clean attributes but preserve important ones
        important_attrs = ['href', 'src', 'alt', 'title']
        for tag in soup.find_all(True):
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in important_attrs:
                    del tag[attr]

        # Get the cleaned HTML
        cleaned_html = str(soup)

        # Remove excessive whitespace
        cleaned_html = ' '.join(cleaned_html.split())

        logger.info(f"HTML cleaned successfully. Reduced length to {len(cleaned_html)} chars")
        return cleaned_html

    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        logger.warning("Returning original HTML content due to cleaning error")
        return html_content


def create_vector_store(documents, embedding_model):
    """Create a FAISS vector store with proper error handling and CPU configuration.

    Args:
        documents (List[Document]): List of documents to index
        embedding_model: The embedding model to use
    Returns:
        FAISS: Configured vector store
    """
    try:
        import faiss
        logger.info("Creating FAISS vector store with CPU configuration")

        # Ensure we're using CPU version
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            logger.info("GPU FAISS available but using CPU for compatibility")

        vector_store = FAISS.from_documents(
            documents,
            embedding_model,
            normalize_L2=True  # L2 normalization for better similarity search
        )

        logger.info(f"Successfully created vector store with {len(documents)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def process_url(url, prompt, max_tokens=14000):
    """Process a URL with RAG: extract HTML, clean, minify, convert to Markdown, chunk, and parse.

    Args:
        url (str): The URL to process
        prompt (str): The user's query prompt
        max_tokens (int): Maximum allowed tokens for the final prompt
    Returns:
        dict: Processing results or error information
    """
    logger.info(f"Starting processing for URL: {url}")
    try:
        # Step 1: Extract HTML with Playwright
        playwright_html_content = extract_html_with_playwright(url)
        if not playwright_html_content:
            raise ValueError("No HTML content extracted from the URL")

        # Step 2: Clean HTML
        html_content = clean_html(playwright_html_content)
        if not html_content:
            raise ValueError("HTML cleaning resulted in empty content")

        # Step 3: Minify HTML with better options
        minified_html = minify(
            html_content,
            remove_comments=True,
            remove_empty_space=True,
            remove_all_empty_space=True,
            remove_optional_attribute_quotes=False
        )
        logger.info(f"Minified HTML. Length reduced from {len(html_content)} to {len(minified_html)} chars")

        # Step 4: Convert to Markdown with better options
        markdown_content = md(
            minified_html,
            heading_style="ATX",
            bullets="-",
            strip=['script', 'style', 'meta', 'link', 'image'],
            default_title=True
        )

        # Clean up markdown content
        markdown_content = "\n".join(line.strip() for line in markdown_content.split("\n") if line.strip())
        logger.info(f"Converted to Markdown. Length: {len(markdown_content)} chars")

        # Save Markdown with timestamp
        output_dir = "markdown_output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        markdown_filename = os.path.join(output_dir, f"extracted_{timestamp}.md")
        with open(markdown_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Saved Markdown to {markdown_filename}")

        # Step 5: Calculate tokens
        tokenizer = tiktoken.get_encoding("cl100k_base")
        markdown_tokens = len(tokenizer.encode(markdown_content))
        logger.info(f"Markdown token count: {markdown_tokens}")

        # Step 6: Chunk the Markdown with improved settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            keep_separator=True,
        )

        chunks = text_splitter.split_text(markdown_content)

        # Filter and clean chunks
        chunks = [
            chunk for chunk in chunks
            if len(chunk.strip()) > 100 and
               not all(c in '[](){}#*-_' for c in chunk.strip())
        ]

        if not chunks:
            raise ValueError("No valid chunks created after filtering")

        logger.info(f"Created {len(chunks)} chunks")
        logger.info(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.2f} chars")

        # Step 7: Create documents with enhanced metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_length": len(chunk),
                    "token_count": len(tokenizer.encode(chunk))
                }
            ) for i, chunk in enumerate(chunks)
        ]

        # Step 8: Create embeddings and vector store
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    # 'normalize_embeddings': True  # Ensure embeddings are normalized
                },
                encode_kwargs={
                    # 'normalize_embeddings': True,
                    'batch_size': 32  # Adjust based on your memory
                }
            )

            # Create vector store with new helper function
            vector_store = create_vector_store(documents, embedding_model)

        except Exception as e:
            logger.error(f"Error in vector store creation: {str(e)}")
            raise Exception(f"Vector store creation failed: {str(e)}")

        # Step 9: Smart retrieval with relevance threshold and error handling
        try:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.5,
                    "fetch_k": 20  # Fetch more candidates initially
                }
            )
            relevant_chunks = retriever.get_relevant_documents(prompt)

            if not relevant_chunks:
                logger.warning("No relevant chunks found, using fallback strategy")
                # Fallback to less strict similarity search
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 3,
                        "score_threshold": 0.3  # Lower threshold
                    }
                )
                relevant_chunks = retriever.get_relevant_documents(prompt)

        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            raise Exception(f"Document retrieval failed: {str(e)}")

        # Sort chunks by relevance score and limit total tokens
        relevant_chunks.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

        # Build context while respecting token limit
        context_chunks = []
        total_tokens = 0
        prompt_tokens = len(tokenizer.encode(prompt))
        max_context_tokens = max_tokens - prompt_tokens - 100  # Buffer of 100 tokens

        for chunk in relevant_chunks:
            chunk_tokens = len(tokenizer.encode(chunk.page_content))
            if total_tokens + chunk_tokens <= max_context_tokens:
                context_chunks.append(chunk.page_content)
                total_tokens += chunk_tokens
            else:
                break

        context = "\n\n---\n\n".join(context_chunks)
        logger.info(f"Selected {len(context_chunks)} relevant chunks within token limit")

        # Step 10: Create augmented prompt
        augmented_prompt = f"""Context Information:
{context}

User Question: {prompt}

Please provide a detailed answer based on the context above."""

        final_tokens = len(tokenizer.encode(augmented_prompt))
        if final_tokens > max_tokens:
            raise ValueError(f"Final prompt exceeds token limit: {final_tokens} > {max_tokens}")

        logger.info(f"Final prompt tokens: {final_tokens}")

        # Step 11: Process with ScrapeGraphAI
        scraper = SmartScraperGraph(
            prompt=augmented_prompt,
            source="",
            config=Config.GRAPH_CONFIG
        )
        final_result = scraper.run()

        return {
            "status": "success",
            "result": final_result,
            "metadata": {
                "total_chunks": len(chunks),
                "used_chunks": len(context_chunks),
                "total_tokens": final_tokens,
                "markdown_file": markdown_filename
            }
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing URL {url}: {error_msg}")
        return {
            "status": "error",
            "error": error_msg,
            "url": url
        }