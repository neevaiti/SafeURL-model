import re
from urllib.parse import urlparse
from tld import get_tld  
from googlesearch import search


def having_ip_address(url):
    """
    Function to check if the input URL contains an IP address.
    Parameters:
    - url: a string representing the URL to be checked.
    Return:
    - 1 if the URL contains an IP address, 0 otherwise.
    """
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)|'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}',  # Ipv6
        url)
    return 1 if match else 0

def abnormal_url(url):
    """
    Function to check if the given URL is abnormal by comparing its hostname with the URL itself.
    
    :param url: The input URL to be checked
    :type url: str
    
    :return: 1 if the hostname is found in the URL, 0 otherwise
    :rtype: int
    """
    hostname = urlparse(url).hostname or ''
    return 1 if re.search(re.escape(hostname), url) else 0

def google_index(url):
    """
    Function to check if a URL is indexed by Google.

    Args:
        url (str): The URL to be checked.

    Returns:
        int: 1 if the URL is indexed, 0 if it is not.
    """
    try:
        site = list(search(url, num=1, stop=1, pause=2))
        return 1 if site else 0
    except Exception as e:
        print(f"Error checking Google index for {url}: {e}")
        return 0

def count_dot(url):
    """
    Count the occurrences of the '.' character in the input URL.
    
    :param url: A string representing the URL
    :return: An integer representing the count of '.' characters in the URL
    """
    return url.count('.')

def count_www(url):
    """
    Count the occurrence of 'www' in the given URL string.

    :param url: The input URL string
    :return: The count of occurrences of 'www' in the URL
    """
    return url.count('www')

def count_atrate(url):
    """
    Count the number of occurrences of '@' in the given URL string.

    Parameters:
    url (str): The input URL string.

    Returns:
    int: The number of occurrences of '@' in the URL.
    """
    return url.count('@')

def no_of_dir(url):
    """
    Calculate the number of directories in the given URL path.

    Parameters:
    url (str): The URL to parse and count the directories from.

    Returns:
    int: The number of directories in the URL path.
    """
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url):
    """
    This function takes a URL as input and returns the number of occurrences of '//' in the path of the URL.
    """
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url):
    """
    Function to check if the given URL is a shortened URL.
    Parameters:
    - url: a string representing the URL to be checked
    Returns:
    - 1 if the URL matches a shortened URL pattern, 0 otherwise
    """
    match = re.search(
        'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
        'tr\.im|link\.zip\.net',
        url)
    return 1 if match else 0

def count_https(url):
    """
    Counts the number of occurrences of 'https' in the given URL.

    :param url: A string representing the URL.
    :return: An integer representing the count of 'https' in the URL.
    """
    return url.count('https')

def count_http(url):
    """
    Count the occurrences of 'http' in the given URL string.

    Parameters:
    url (str): The URL string to search for occurrences of 'http'.

    Returns:
    int: The number of occurrences of 'http' in the URL string.
    """
    return url.count('http')

def count_per(url):
    """
    Count the occurrences of '%' in the given URL string and return the count.
    Parameters:
    url (str): The URL string to count '%' occurrences in.
    Returns:
    int: The count of '%' occurrences in the URL string.
    """
    return url.count('%')

def count_ques(url):
    """
    Count the number of question marks in the given URL string.

    Parameters:
    url (str): The URL string to be checked for question marks.

    Returns:
    int: The number of question marks in the URL string.
    """
    return url.count('?')

def count_hyphen(url):
    """
    Count the number of hyphens in the given URL string.

    Args:
        url (str): The URL string to count hyphens in.

    Returns:
        int: The number of hyphens in the URL string.
    """
    return url.count('-')

def count_equal(url):
    """
    Count the number of occurrences of the '=' character in the given URL string.

    Parameters:
    url (str): The URL string to count '=' occurrences in.

    Returns:
    int: The number of occurrences of '=' in the URL string.
    """
    return url.count('=')

def url_length(url):
    """
    Calculate the length of the input URL string and return the result.
    """
    return len(str(url))

def hostname_length(url):
    """
    Calculate the length of the hostname extracted from the given URL.

    Args:
        url (str): The input URL from which the hostname will be extracted.

    Returns:
        int: The length of the extracted hostname.
    """
    return len(urlparse(url).netloc)

def suspicious_words(url):
    """
    Check if the given URL contains suspicious words related to phishing or fraudulent activities. 
    Parameters:
    - url: a string representing the URL to be checked
    Return:
    - 1 if the URL contains any of the suspicious words, 0 otherwise
    """
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def digit_count(url):
    """
    Calculate the number of digits in the given URL string.
    
    :param url: A string representing the URL.
    :return: An integer representing the count of digits in the URL.
    """
    return sum(c.isdigit() for c in url)

def letter_count(url):
    """
    This function takes a URL as input and returns the count of letters in the URL.
    """
    return sum(c.isalpha() for c in url)

def fd_length(url):
    """
    Calculate the length of the second segment in the path of the given URL.

    :param url: The input URL
    :type url: str
    :return: The length of the second segment in the URL path, or 0 if there is no second segment
    :rtype: int
    """
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(url):
    """
    Calculate the length of the top-level domain (TLD) of the given URL.

    Parameters:
    url (str): The URL for which the TLD length needs to be calculated.

    Returns:
    int: The length of the TLD if it exists, -1 otherwise.
    """
    try:
        tld = get_tld(url, fail_silently=True)
        return len(tld) if tld else -1
    except:
        return -1
    
    

def extract_features(url):
    """
    Function to extract features from a given URL.

    Parameters:
    url (str): The URL from which to extract features.

    Returns:
    dict: A dictionary containing various features extracted from the URL.
    """
    features = {
        'use_of_ip': having_ip_address(url),
        'abnormal_url': abnormal_url(url),
        'count.': count_dot(url),
        'count-www': count_www(url),
        'count@': count_atrate(url),
        'count_dir': no_of_dir(url),
        'count_embed_domian': no_of_embed(url),
        'short_url': shortening_service(url),
        'count-https': count_https(url),
        'count-http': count_http(url),
        'count%': count_per(url),
        'count?': count_ques(url),
        'count-': count_hyphen(url),
        'count=': count_equal(url),
        'url_length': url_length(url),
        'hostname_length': hostname_length(url),
        'sus_url': suspicious_words(url),
        'fd_length': fd_length(url),
        'tld_length': tld_length(url),
        'count-digits': digit_count(url),
        'count-letters': letter_count(url),
    }
    return features