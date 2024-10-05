def clean_url(url: str) -> str:
    """
    Clean the input URL by removing 'http://', 'https://', 'www.', and trailing slashes.
    
    Args:
    url (str): The input URL to clean.
    
    Returns:
    str: The cleaned URL.
    """
    url = str(url)  # Convert the URL to a string if it is not already a string
    
    # Remove 'http://' and 'https://' schemes
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    
    # Remove 'www.' if present
    if url.startswith('www.'):
        url = url[4:]
    
    # Remove trailing slashes
    if url.endswith('/'):
        url = url.rstrip('/')
        
    return url