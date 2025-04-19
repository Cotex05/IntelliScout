def format_json_as_markdown(data):
    """
    Format JSON data as beautiful markdown.
    - Keys become subheadings (## level)
    - String values become paragraphs
    - Array values become bullet points
    - Nested objects are recursively formatted
    
    Args:
        data: The data to format (dict, list, or simple value)
    Returns:
        Formatted markdown string
    """
    if isinstance(data, dict):
        md_parts = []
        
        # Loop through dictionary items and format each
        for key, value in data.items():
            # Make the key a subheading (with title case for readability)
            key_title = str(key).replace("_", " ").title()
            md_parts.append(f"### {key_title}")
            
            # Format the value based on its type
            if isinstance(value, dict):
                # For nested dictionaries, indent and process recursively
                nested_md = format_json_as_markdown(value)
                md_parts.append(nested_md)
            elif isinstance(value, list):
                # For lists, create bullet points
                if not value:
                    md_parts.append("*No items found*\n")
                else:
                    for item in value:
                        if isinstance(item, dict):
                            # For a list of objects, format each as its own section
                            md_parts.append(format_json_as_markdown(item))
                        else:
                            # For simple values, create bullet points
                            md_parts.append(f"* {item}")
                    md_parts.append("")  # Add spacing after list
            else:
                # For simple values, create a paragraph
                if value is None:
                    md_parts.append("*Not available*\n")
                else:
                    md_parts.append(f"{value}\n")
        
        return "\n".join(md_parts)
    
    elif isinstance(data, list):
        md_parts = []
        
        # Format each item in the list
        for item in data:
            if isinstance(item, dict):
                md_parts.append(format_json_as_markdown(item))
            else:
                md_parts.append(f"* {item}")
        
        return "\n".join(md_parts)
    
    else:
        # Simple value, return as is
        return str(data)


def format_response_for_display(result, output_format):
    """
    Format the response for display based on the output format
    
    Args:
        result: The result to format
        output_format (str): The desired output format ("json" or "markdown")
        
    Returns:
        tuple: (display_data, metadata)
    """
    metadata = None
    display_data = result
    
    # Extract metadata if available
    if isinstance(result, dict):
        if "metadata" in result:
            metadata = result.get("metadata", {})
        
        if output_format.lower() == "markdown" and "result" in result:
            # For markdown display, extract the result field
            if isinstance(result["result"], dict):
                display_data = format_json_as_markdown(result["result"])
            else:
                display_data = result["result"]
        elif output_format.lower() == "json":
            # For JSON display, use the entire result
            display_data = result
    
    return display_data, metadata 