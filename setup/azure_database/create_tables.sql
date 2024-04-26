CREATE TABLE IF NOT EXISTS Model_predict (
    url_id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    type TEXT,
    DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)


CREATE TABLE IF NOT EXISTS Model_results (
    FOREIGN KEY(url_id) REFERENCES Model_predict(url_id),
    url TEXT,
    result TEXT
)


CREATE TABLE IF NOT EXISTS Model_training (
    url TEXT,
    type INTEGER,
    use_of_ip INTEGER,
    abnormal_url INTEGER,
    count-wwww INTEGER,
    count-point INTEGER,
    count@ INTEGER,
    count-https INTEGER,
    count-http INTEGER,
    count% INTEGER,
    count? INTEGER,
    count- INTEGER,
    count= INTEGER,
    count_dir INTEGER,
    count_embed_domain INTEGER,
    short_url INTEGER,
    url_length INTEGER,
    hostname_length INTEGER,
    sus_url INTEGER,
    count-digits INTEGER,
    count-letters INTEGER,
    fd_length INTEGER,
    tld_length INTEGER
)


CREATE TABLE IF NOT EXISTS List_url (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    type TEXT
)