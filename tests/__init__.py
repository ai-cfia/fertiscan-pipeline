import Levenshtein
import requests

def levenshtein_similarity(str1, str2):
    if str1 is None:
        str1 = ''
    if str2 is None:
        str2 = ''
    # Calculate the Levenshtein distance
    distance = Levenshtein.distance(str1, str2)
    
    # Determine the maximum possible length
    max_len = max(len(str1), len(str2))
    
    if max_len == 0:
        return 100.0  # If both strings are empty, they are identical
    
    # Calculate the similarity as a percentage
    similarity_percentage = (1 - (distance / max_len)) * 100
    
    return similarity_percentage


def curl_file(url:str, path: str): # pragma: no cover
    """
    Pull a file from an URL and save its content.
    """
    data = requests.get(url).content
    with open(path, 'wb') as handler:
        handler.write(data)  