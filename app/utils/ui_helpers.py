import os
import streamlit as st
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def display_screenshot(url):
    """
    Display a screenshot for the given URL if it exists
    
    Args:
        url (str): The URL of the webpage
        
    Returns:
        bool: True if screenshot was displayed, False otherwise
    """
    try:
        # Create a filename from the URL (sanitized)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")
        filename = f"{domain}.png"
        
        # Check if screenshot exists
        screenshots_dir = Path("screenshots")
        screenshot_path = screenshots_dir / filename
        
        if not screenshot_path.exists():
            logger.info(f"Screenshot not found: {screenshot_path}")
            return False
        
        # Display the screenshot
        try:
            img = Image.open(screenshot_path)
            st.subheader("Website Preview")
            st.image(img, width=300, use_container_width=False)
            
            # Add download button for full-size image
            with open(screenshot_path, "rb") as file:
                btn = st.download_button(
                    label="Download Full Screenshot",
                    data=file,
                    file_name=f"{domain}_screenshot.png",
                    mime="image/png"
                )
            
            return True
        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in display_screenshot: {str(e)}")
        return False

def create_screenshots_dir():
    """
    Ensure the screenshots directory exists
    
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs("screenshots", exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating screenshots directory: {str(e)}")
        return False

def display_info_box(title, content, type="info"):
    """
    Display an information box with custom styling
    
    Args:
        title (str): The title of the info box
        content (str): The content to display
        type (str): The type of box (info, warning, error, success)
    """
    if type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif type == "error":
        st.error(f"**{title}**\n\n{content}")
    elif type == "success":
        st.success(f"**{title}**\n\n{content}")
    else:
        st.write(f"**{title}**\n\n{content}") 