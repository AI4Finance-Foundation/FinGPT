import urllib

def url_encode_string(input_string):
    encoded_string = urllib.parse.quote(input_string)
    return encoded_string