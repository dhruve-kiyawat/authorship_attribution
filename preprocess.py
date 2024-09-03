from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from sacremoses import MosesTokenizer, MosesPunctNormalizer

# Removal of punctuation
import re 
from spacy.lang.hi import STOP_WORDS as STOP_WORDS_HI

def remove_punctuations(text):
    # Define Hindi punctuation marks
    hindi_punctuation = "।,;:؟!‘’“”-''–?—.()…*&¬\""
    # Create a regex pattern to match any of the defined punctuation marks
    pattern = f"[{re.escape(hindi_punctuation)}]"

    # Replace punctuation with an empty string
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text

# Normalize
def normalize_text(text):
    # Unicode Normalization
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer("hi")
    
    # Remove special characters and punctuations
    tokenizer = MosesTokenizer(lang='hi')
    normalizer = MosesPunctNormalizer()
    text = normalizer.normalize(text)
    
    # Tokenize and rejoin to remove unnecessary spaces
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

# Tokenization
def tokenize_text(text):
    # Word tokenization using Indic NLP
    tokens = list(indic_tokenize.trivial_tokenize(text))
    return tokens

# Stopwords removal
def stopword_remover(text):
    non_stop_words = [word for word in text if word not in set(STOP_WORDS_HI) ]
    return non_stop_words
