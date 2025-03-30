import json

def process_chroma_metadata(metadata):
    """
    Process ChromaDB metadata by converting string-encoded lists and dicts back to their original form.
    
    Args:
        metadata (dict): Metadata from ChromaDB
        
    Returns:
        dict: Processed metadata
    """
    processed = {}
    
    for key, value in metadata.items():
        # Try to convert pipe-separated strings back to lists
        if isinstance(value, str) and "|" in value:
            try:
                processed[key] = value.split("|")
            except:
                processed[key] = value
        # Try to convert JSON strings back to dicts
        elif isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                processed[key] = json.loads(value)
            except:
                processed[key] = value
        else:
            processed[key] = value
    
    return processed


def format_sources_for_display(sources, max_sources=3):
    """
    Format source information for display.
    
    Args:
        sources (list): List of source dictionaries
        max_sources (int): Maximum number of sources to include
        
    Returns:
        str: Formatted source information
    """
    if not sources:
        return "No sources available."
    
    formatted = "\nSources:\n"
    
    for i, source in enumerate(sources[:max_sources]):
        formatted += f"{i+1}. {source.get('title', 'Unknown')}\n"
        formatted += f"   URL: {source.get('url', 'N/A')}\n"
        
        if source.get('section'):
            formatted += f"   Section: {source['section']}\n"
            
        if i < len(sources[:max_sources]) - 1:
            formatted += "\n"
    
    if len(sources) > max_sources:
        formatted += f"\n...and {len(sources) - max_sources} more sources."
    
    return formatted