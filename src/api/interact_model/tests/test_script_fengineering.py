from api.interact_model.utils.script_fengineering import extract_features

def test_extract_features_basic():
    url = "http://www.example.com"
    features = extract_features(url)
    
    assert isinstance(features, dict)
    assert 'use_of_ip' in features
    assert features['use_of_ip'] == 0
    assert 'abnormal_url' in features
    assert 'count_www' in features
    assert features['count_www'] == 1

def test_extract_features_ip():
    url = "http://192.168.1.1"
    features = extract_features(url)
    
    assert features['use_of_ip'] == 1
    assert features['abnormal_url'] == 1

def test_extract_features_long_url():
    url = "http://www.example.com/" + "a" * 100
    features = extract_features(url)
    
    assert features['url_length'] > 100
    assert features['short_url'] == 0

def test_extract_features_short_url():
    url = "http://bit.ly/abc123"
    features = extract_features(url)
    
    assert features['short_url'] == 1

def test_extract_features_special_chars():
    url = "http://www.example.com/path?param=value#fragment"
    features = extract_features(url)
    
    assert features['count_point'] >= 1
    assert features['count_question'] == 1
    assert features['count_equal'] == 1
    assert features['count_dir'] >= 1

def test_extract_features_suspicious():
    url = "http://example-bank.com.suspicious.com"
    features = extract_features(url)
    
    assert features['sus_url'] == 1

def test_extract_features_https():
    url = "https://www.example.com"
    features = extract_features(url)
    
    assert features['count_https'] == 1
    assert features['count_http'] == 1

def test_extract_features_http():
    url = "http://www.example.com"
    features = extract_features(url)
    
    assert features['count_https'] == 0
    assert features['count_http'] == 1

def test_extract_features_percent():
    url = "http://www.example.com/%20test"
    features = extract_features(url)
    
    assert features['count_percent'] == 1

def test_extract_features_hyphen():
    url = "http://www.example-test.com"
    features = extract_features(url)
    
    assert features['count_dash'] == 1

def test_extract_features_equal():
    url = "http://www.example.com?param=value"
    features = extract_features(url)
    
    assert features['count_equal'] == 1

def test_extract_features_digits():
    url = "http://www.example123.com"
    features = extract_features(url)
    
    assert features['count_digits'] == 3

def test_extract_features_letters():
    url = "http://www.example.com"
    features = extract_features(url)
    
    assert features['count_letters'] == 17

def test_extract_features_tld_length():
    url = "http://www.example.com"
    features = extract_features(url)
    
    assert features['tld_length'] == 3

def test_extract_features_fd_length():
    url = "http://www.example.com/path/to/page"
    features = extract_features(url)
    
    assert features['fd_length'] == len("path")
