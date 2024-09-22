CREATE TABLE IF NOT EXISTS Model_predict (
    url_id SERIAL PRIMARY KEY,
    url TEXT,
    type TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Model_results (
    result_id SERIAL PRIMARY KEY,
    url_id INTEGER REFERENCES Model_predict(url_id),
    url TEXT,
    result TEXT
);

CREATE TABLE IF NOT EXISTS Model_training (
    id SERIAL PRIMARY KEY,
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
    id SERIAL PRIMARY KEY,
    url TEXT,
    type TEXT
);

CREATE TABLE IF NOT EXISTS Log (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT,
    message TEXT
);

ALTER TABLE List_url ADD CONSTRAINT unique_url UNIQUE (url);
