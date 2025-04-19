import re
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def clean_html(html_content):
    """
    Clean HTML content by removing unnecessary elements and formatting
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        str: Cleaned HTML content
    """
    try:
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "svg", "noscript", "iframe"]):
            script.extract()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Remove excessive newlines (more than 2 in a row)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        return html_content  # Return original content in case of error

def extract_main_content(html_content):
    """
    Extract the main content from an HTML page
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        str: Main content text
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove header, footer, nav, and similar elements
        for element in soup.find_all(['header', 'footer', 'nav', 'aside']):
            element.extract()
        
        # Find main content containers
        content_tags = soup.find_all(['article', 'main', 'div.content', 'div.main'])
        
        if content_tags:
            # Use the first found content tag
            main_content = content_tags[0].get_text(separator='\n', strip=True)
        else:
            # If no content tags found, use the body
            main_content = soup.find('body').get_text(separator='\n', strip=True)
        
        # Clean up the extracted text
        main_content = re.sub(r'\n{3,}', '\n\n', main_content)
        
        return main_content
    except Exception as e:
        logger.error(f"Error extracting main content: {str(e)}")
        # Fallback to regular cleaning if extraction fails
        return clean_html(html_content)

def truncate_content(content, max_length=8000):
    """
    Truncate content to a maximum length while preserving whole paragraphs
    
    Args:
        content (str): Text content to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated content
    """
    if len(content) <= max_length:
        return content
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    result = []
    current_length = 0
    
    for paragraph in paragraphs:
        if current_length + len(paragraph) + 2 <= max_length:  # +2 for '\n\n'
            result.append(paragraph)
            current_length += len(paragraph) + 2
        else:
            break
    
    return '\n\n'.join(result) 