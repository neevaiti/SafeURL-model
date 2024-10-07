from api.interact_model.utils.clean_url import clean_url

def test_clean_url_with_protocol():
    assert clean_url("http://www.example.com") == "example.com"
    assert clean_url("https://www.example.com") == "example.com"
    assert clean_url("http://example.com") == "example.com"
    assert clean_url("https://example.com") == "example.com"

def test_clean_url_with_www():
    assert clean_url("www.example.com") == "example.com"
    assert clean_url("http://www.example.com") == "example.com"
    assert clean_url("https://www.example.com") == "example.com"

def test_clean_url_with_subdomain():
    assert clean_url("sub.example.com") == "sub.example.com"
    assert clean_url("http://sub.example.com") == "sub.example.com"
    assert clean_url("https://sub.example.com") == "sub.example.com"
    assert clean_url("www.sub.example.com") == "sub.example.com"
    assert clean_url("http://www.sub.example.com") == "sub.example.com"
    assert clean_url("https://www.sub.example.com") == "sub.example.com"

def test_clean_url_with_path():
    assert clean_url("example.com/path/to/page") == "example.com/path/to/page"
    assert clean_url("http://example.com/path/to/page") == "example.com/path/to/page"
    assert clean_url("https://example.com/path/to/page") == "example.com/path/to/page"
    assert clean_url("www.example.com/path/to/page") == "example.com/path/to/page"
    assert clean_url("http://www.example.com/path/to/page") == "example.com/path/to/page"
    assert clean_url("https://www.example.com/path/to/page") == "example.com/path/to/page"

def test_clean_url_with_query_params():
    assert clean_url("example.com?param=value") == "example.com?param=value"
    assert clean_url("http://example.com?param=value") == "example.com?param=value"
    assert clean_url("https://example.com?param=value") == "example.com?param=value"
    assert clean_url("www.example.com?param=value") == "example.com?param=value"
    assert clean_url("http://www.example.com?param=value") == "example.com?param=value"
    assert clean_url("https://www.example.com?param=value") == "example.com?param=value"

def test_clean_url_with_fragment():
    assert clean_url("example.com#fragment") == "example.com#fragment"
    assert clean_url("http://example.com#fragment") == "example.com#fragment"
    assert clean_url("https://example.com#fragment") == "example.com#fragment"
    assert clean_url("www.example.com#fragment") == "example.com#fragment"
    assert clean_url("http://www.example.com#fragment") == "example.com#fragment"
    assert clean_url("https://www.example.com#fragment") == "example.com#fragment"

def test_clean_url_with_trailing_slash():
    assert clean_url("example.com/") == "example.com"
    assert clean_url("http://example.com/") == "example.com"
    assert clean_url("https://example.com/") == "example.com"
    assert clean_url("www.example.com/") == "example.com"
    assert clean_url("http://www.example.com/") == "example.com"
    assert clean_url("https://www.example.com/") == "example.com"

def test_clean_url_complex():
    assert clean_url("https://sub.example.co.uk/path?param=value#fragment") == "sub.example.co.uk/path?param=value#fragment"
    assert clean_url("http://www.sub.example.co.uk/path?param=value#fragment") == "sub.example.co.uk/path?param=value#fragment"
    assert clean_url("https://www.sub.example.co.uk/path?param=value#fragment") == "sub.example.co.uk/path?param=value#fragment"
    assert clean_url("www.sub.example.co.uk/path?param=value#fragment") == "sub.example.co.uk/path?param=value#fragment"
    assert clean_url("sub.example.co.uk/path?param=value#fragment") == "sub.example.co.uk/path?param=value#fragment"
