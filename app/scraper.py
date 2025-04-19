import os
import time
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright
from htmlmin import minify
from markdownify import markdownify as md
from scrapegraphai.graphs import SmartScraperGraph, SearchGraph
from config import Config, logger
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from urllib.parse import urlparse

def save_screenshot(page, url):
    """Save a screenshot of the opened page to a directory."""
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    # Create filename based on domain and timestamp
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace("www.", "")
    
    # Save both domain-specific and generic screenshot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    domain_screenshot = os.path.join(screenshot_dir, f"{domain}.png")
    generic_screenshot = os.path.join(screenshot_dir, f"screenshot.png")
    
    try:
        # Take and save full page screenshot
        page.screenshot(path=domain_screenshot, full_page=True)
        # Also save as generic screenshot.png for backward compatibility
        page.screenshot(path=generic_screenshot, full_page=True)
        logger.info(f"Saved screenshot to {domain_screenshot}")
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
                    wait_until="domcontentloaded",
                    timeout=45000
                )
                
                if not response or not response.ok:
                    raise Exception(f"Failed to get response from {url}. Status: {response.status if response else 'No response'}")
                
                # Wait for key elements to be present
                try:
                    # Wait for body to be loaded
                    page.wait_for_selector('body', timeout=10000)
                    
                    # Additional waits for SPAs
                    page.wait_for_load_state('domcontentloaded')
                    # page.wait_for_load_state('networkidle')
                    
                    # Wait a bit more for dynamic content
                    page.wait_for_timeout(2000)
                    
                    # Scroll to bottom to trigger lazy loading
                    # page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
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

def process_url(url, prompt, max_tokens=10000, use_direct_url=False, output_format="markdown"):
    """Process a URL: extract HTML, clean, minify, convert to Markdown, and parse.
    
    Args:
        url (str): The URL to process
        prompt (str): The query prompt
        max_tokens (int): Maximum tokens to use
        use_direct_url (bool): If True, passes URL directly to model. If False, processes content first.
        output_format (str): The desired output format ("markdown" or "json")
    """
    logger.info(f"Starting processing for URL: {url} (Direct URL: {use_direct_url})")
    
    try:
        if use_direct_url:
            # Direct URL mode: Pass URL directly to SmartScraperGraph
            logger.info("Using direct URL mode")
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,  # Pass URL directly
                config=Config.GRAPH_CONFIG
            )
            final_result = scraper.run()
            
            if output_format.lower() == "json":
                return {
                    "status": "success",
                    "result": final_result,
                    "metadata": {
                        "mode": "direct_url",
                        "url": url
                    }
                }
            else:
                return final_result
        
        # Content processing mode
        logger.info("Using content processing mode")
        
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
        markdown_filename = os.path.join(output_dir, f"extracted_markdown.md")
        with open(markdown_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"Saved Markdown to {markdown_filename}")

        # Step 5: Calculate tokens and prepare for chunking
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Reserve tokens for the prompt and system message
        PROMPT_TOKENS = len(tokenizer.encode(prompt))
        SYSTEM_RESERVE = 500  # Reserve tokens for system message and overhead
        RESPONSE_RESERVE = 1000  # Reserve tokens for model's response
        
        # Calculate available tokens for content
        AVAILABLE_TOKENS = max_tokens - PROMPT_TOKENS - SYSTEM_RESERVE - RESPONSE_RESERVE
        
        if AVAILABLE_TOKENS < 500:  # Minimum reasonable context size
            raise ValueError(f"Not enough tokens available. Prompt is too long ({PROMPT_TOKENS} tokens)")
            
        logger.info(f"Token budget - Total: {max_tokens}, Available: {AVAILABLE_TOKENS}, Prompt: {PROMPT_TOKENS}")

        # Step 6: Chunk the Markdown with token-aware sizing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better token control
            chunk_overlap=50,  # Reduced overlap
            length_function=lambda x: len(tokenizer.encode(x)),  # Use actual token count
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            keep_separator=True,
        )
        
        chunks = text_splitter.split_text(markdown_content)
        
        # Filter and clean chunks
        chunks = [
            chunk for chunk in chunks 
            if len(chunk.strip()) > 50 and  # Reduced minimum size
            not all(c in '[](){}#*-_' for c in chunk.strip())
        ]
        
        if not chunks:
            raise ValueError("No valid chunks created after filtering")
            
        logger.info(f"Created {len(chunks)} initial chunks")

        # Step 7: Smart chunk selection with token awareness
        def get_relevant_chunks(chunks, query, available_tokens):
            # Convert query to lowercase for case-insensitive matching
            query_words = set(query.lower().split())
            
            # Score and filter chunks
            chunk_scores = []
            current_tokens = 0
            
            for chunk in chunks:
                # Calculate token count for this chunk
                chunk_tokens = len(tokenizer.encode(chunk))
                
                # Score the chunk based on keyword matches
                chunk_lower = chunk.lower()
                score = sum(1 for word in query_words if word in chunk_lower)
                
                # Add to candidates if it has matches
                if score > 0:
                    chunk_scores.append((chunk, score, chunk_tokens))
            
            # Sort by relevance score
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select chunks while respecting token limit
            selected_chunks = []
            total_tokens = 0
            
            for chunk, score, chunk_tokens in chunk_scores:
                if total_tokens + chunk_tokens <= available_tokens:
                    selected_chunks.append(chunk)
                    total_tokens += chunk_tokens
                else:
                    break
            
            logger.info(f"Selected {len(selected_chunks)} chunks using {total_tokens} tokens")
            return selected_chunks, total_tokens

        # Get relevant chunks within token limit
        selected_chunks, used_tokens = get_relevant_chunks(chunks, prompt, AVAILABLE_TOKENS)
        
        if not selected_chunks:
            # Fallback: take the most important chunks that fit
            logger.warning("No keyword matches found, using initial chunks up to token limit")
            selected_chunks, used_tokens = get_relevant_chunks(chunks[:5], "", AVAILABLE_TOKENS)
        
        # Join selected chunks with separators
        source_content = "\n\n---\n\n".join(selected_chunks)
        
        # Process with SmartScraperGraph using processed content
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=source_content,  # Use processed content
            config=Config.GRAPH_CONFIG
        )
        final_result = scraper.run()

        if output_format.lower() == "json":
            return {
                "status": "success",
                "result": final_result,
                "metadata": {
                    "mode": "processed_content",
                    "total_chunks": len(chunks),
                    "used_chunks": len(selected_chunks),
                    "prompt_tokens": PROMPT_TOKENS,
                    "content_tokens": used_tokens,
                    "total_tokens": PROMPT_TOKENS + used_tokens + SYSTEM_RESERVE,
                    "markdown_file": markdown_filename,
                    "url": url
                }
            }
        else:
            return final_result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing URL {url}: {error_msg}")
        if output_format.lower() == "json":
            return {
                "status": "error",
                "error": error_msg,
                "url": url,
                "mode": "direct_url" if use_direct_url else "processed_content"
            }
        else:
            return f"Error processing URL: {error_msg}"

def process_search(search_prompt, max_tokens=10000, output_format="json"):
    """Process a search query using SearchGraph.
    
    Args:
        search_prompt (str): The search query
        max_tokens (int): Maximum tokens to use
        output_format (str): The desired output format ("markdown" or "json")
    
    Returns:
        dict or str: Search results in the specified format
    """
    logger.info(f"Starting search processing with query: '{search_prompt}'")
    logger.info(f"Using model: {Config.MODEL_ID} with max_tokens: {max_tokens}")
    
    try:
        # Initialize SearchGraph with the prompt and config
        search_scraper = SearchGraph(
            prompt=search_prompt,
            config=Config.GRAPH_CONFIG
        )
        
        # Log the actual configuration being used
        logger.info(f"Search config: {Config.GRAPH_CONFIG}")
        
        # Run the search
        search_result = search_scraper.run()
        logger.info(f"Search completed successfully")
        
        # Format the response based on the output format
        if output_format.lower() == "json":
            return {
                "status": "success",
                "result": search_result,
                "metadata": {
                    "mode": "search",
                    "query": search_prompt,
                    "model": Config.MODEL_ID,
                    "max_tokens": max_tokens
                }
            }
        else:
            return search_result
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing search '{search_prompt}': {error_msg}")
        
        if output_format.lower() == "json":
            return {
                "status": "error",
                "error": error_msg,
                "query": search_prompt,
                "mode": "search"
            }
        else:
            return f"Error processing search: {error_msg}"