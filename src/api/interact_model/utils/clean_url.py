def clean_url(url: str) -> str:
    """
    Nettoie l'URL d'entrée en supprimant 'http://', 'https://', 'www.', et les slashs de fin.
    
    Args:
    url (str): L'URL d'entrée à nettoyer.
    
    Returns:
    str: L'URL nettoyée.
    """
    url = str(url)  # Convertir l'URL en chaîne de caractères si ce n'est pas déjà une chaîne
    
    # Supprimer les schémas 'http://' et 'https://'
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    
    # Supprimer 'www.' si présent
    if url.startswith('www.'):
        url = url[4:]
    
    # Supprimer les slashs de fin
    if url.endswith('/'):
        url = url.rstrip('/')
        
    return url
