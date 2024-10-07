def clean_url(url):
    """
    Function to clean the input URL by removing 'http://', 'https://', 'www.', and trailing slashes.
    
    Args:
    url (str): The input URL to be cleaned.
    
    Returns:
    str: The cleaned URL.
    """
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    if 'www.' in url:
        url = url.replace('www.', '')
    if url.endswith('/'):
        url = url[:-1]
        
    return url
