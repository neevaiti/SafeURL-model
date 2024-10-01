CREATE TABLE IF NOT EXISTS Model_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    type INTEGER,
    use_of_ip INTEGER,
    abnormal_url INTEGER,
    count_www INTEGER,
    count_point INTEGER,
    count_at INTEGER,
    count_https INTEGER,
    count_http INTEGER,
    count_percent INTEGER,
    count_question INTEGER,
    count_dash INTEGER,
    count_equal INTEGER,
    count_dir INTEGER,
    count_embed_domain INTEGER,
    short_url INTEGER,
    url_length INTEGER,
    hostname_length INTEGER,
    sus_url INTEGER,
    count_digits INTEGER,
    count_letters INTEGER,
    fd_length INTEGER,
    tld_length INTEGER
);

CREATE TABLE IF NOT EXISTS List_url (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    type TEXT,
    CONSTRAINT unique_url UNIQUE (url)
);

CREATE TABLE IF NOT EXISTS Log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT,
    message TEXT
);