import re
from urllib.parse import urlparse

def is_valid_url(url):
    """
    Validate if a string is a properly formatted URL
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url or not url.strip():
        return False
        
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

def is_valid_search_query(query):
    """
    Validate if a search query is not empty
    
    Args:
        query (str): Search query to validate
        
    Returns:
        bool: True if query is valid, False otherwise
    """
    return bool(query and query.strip()) 