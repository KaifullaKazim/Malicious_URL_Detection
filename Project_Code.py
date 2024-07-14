import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
from tld import get_tld
import re
import csv

# Load the dataset
data = pd.read_csv('malicious_phish.csv')
data['type'] = data['type'].fillna('unknown')
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data[['url']], data['type'], test_size=0.2, random_state=42)



# Mapping dictionary for 'type' to 'Result'
type_to_result = {
    'benign': 0,
    'malicious': 1,
    'phishing': -1,
    'defacement': 2,
    'malware': -2,
    'unknown': 3
}

# Create 'Result' column based on 'type'
data['Result'] = data['type'].map(type_to_result) # Now 'data' is a DataFrame and map will work

# Feature extraction functions
def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def digit_count(url):
    return sum(c.isdigit() for c in url)

def letter_count(url):
    return sum(c.isalpha() for c in url)

def no_of_dir(url):
    return urlparse(url).path.count('/')

def having_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'  # IPv4
        r'((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)' # IPv4 in hexadecimal
        r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # IPv6
    if match:
        return -1  # use of IP within URL
    else:
        return 1  # IP not found within URL

def shortening_service(url):
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      r'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1  # indicates that URL is shortened
    else:
        return 1  # URL not shortened

def is_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return 1 if youtube_regex.match(url) else -1

# Adding new features to the dataset
data['url_length'] = data['url'].apply(lambda i: len(str(i)))
data['hostname_length'] = data['url'].apply(lambda i: len(urlparse(i).netloc))
data['fd_length'] = data['url'].apply(lambda i: fd_length(i))
data['tld'] = data['url'].apply(lambda i: get_tld(i, fail_silently=True))
data['tld_length'] = data['tld'].apply(lambda i: tld_length(i))
data.drop("tld", axis=1, inplace=True)
data['count-'] = data['url'].apply(lambda i: i.count('-'))
data['count@'] = data['url'].apply(lambda i: i.count('@'))
data['count?'] = data['url'].apply(lambda i: i.count('?'))
data['count%'] = data['url'].apply(lambda i: i.count('%'))
data['count.'] = data['url'].apply(lambda i: i.count('.'))
data['count='] = data['url'].apply(lambda i: i.count('='))
data['count-http'] = data['url'].apply(lambda i: i.count('http'))
data['count-https'] = data['url'].apply(lambda i: i.count('https'))
data['count-www'] = data['url'].apply(lambda i: i.count('www'))
data['count-digits'] = data['url'].apply(lambda i: digit_count(i))
data['count-letters'] = data['url'].apply(lambda i: letter_count(i))
data['count_dir'] = data['url'].apply(lambda i: no_of_dir(i))
data['use_of_ip'] = data['url'].apply(lambda i: having_ip_address(i))
data['short_url'] = data['url'].apply(lambda i: shortening_service(i))
data['short_url'] = data['url'].apply(lambda i: shortening_service(i))
# Predictor and target variables
x = data[['hostname_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?', 'count%', 'count.', 'count=', 'count-http', 'count-https', 'count-www', 'count-digits', 'count-letters', 'count_dir', 'use_of_ip', 'short_url']]
y = data['Result']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Train a Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# Function to extract features from a single URL
def extract_features(url):
    features = {}
    features['hostname_length'] = len(urlparse(url).netloc)
    features['fd_length'] = fd_length(url)
    tld = get_tld(url, fail_silently=True)
    features['tld_length'] = tld_length(tld)
    features['count-'] = url.count('-')
    features['count@'] = url.count('@')
    features['count?'] = url.count('?')
    features['count%'] = url.count('%')
    features['count.'] = url.count('.')
    features['count='] = url.count('=')
    features['count-http'] = url.count('http')
    features['count-https'] = url.count('https')
    features['count-www'] = url.count('www')
    features['count-digits'] = digit_count(url)
    features['count-letters'] = letter_count(url)
    features['count_dir'] = no_of_dir(url)
    features['use_of_ip'] = having_ip_address(url)
    features['short_url'] = shortening_service(url)
    return pd.DataFrame([features])

# Function to predict URL type
def predict_url_type(url):
    new_data = extract_features(url)
    prediction = rfc.predict(new_data)
    result_to_type = {0: 'benign', 1: 'malicious', -1: 'phishing', 2: 'defacement', -2: 'malware'}
    return result_to_type[prediction[0]]

# Take a URL input from the user
new_url = input("Enter a URL: ")

# Predict the type of the URL
url_type = predict_url_type(new_url)
print(f"The URL is predicted to be: {url_type}")

# Optionally, add the new URL to the dataset
new_entry = {
    'url': new_url,
    'hostname_length': len(urlparse(new_url).netloc),
    'fd_length': fd_length(new_url),
    'tld_length': tld_length(get_tld(new_url, fail_silently=True)),
    'count-': new_url.count('-'),
    'count@': new_url.count('@'),
    'count?': new_url.count('?'),
    'count%': new_url.count('%'),
    'count.': new_url.count('.'),
    'count=': new_url.count('='),
    'count-http': new_url.count('http'),
    'count-https': new_url.count('https'),
    'count-www': new_url.count('www'),
    'count-digits': digit_count(new_url),
    'count-letters': letter_count(new_url),
    'count_dir': no_of_dir(new_url),
    'use_of_ip': having_ip_address(new_url),
    'short_url': shortening_service(new_url),
}


