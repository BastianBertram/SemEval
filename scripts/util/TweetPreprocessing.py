'''
Created on 23.01.2017

@author: bastianbertram
'''


import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
    
import re 
import regex as reg
import pandas as pd 


from nltk.corpus import stopwords

stop = stopwords.words('english')


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [ emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
    ]
    
tokens_re = reg.compile(r'('+'|'.join(regex_str)+')', reg.VERBOSE | reg.IGNORECASE)
emoticon_re = reg.compile(r'^'+emoticons_str+'$', reg.VERBOSE | reg.IGNORECASE)

    
    
def remove_url(tweet):
    return re.sub(r"http\S+", "", tweet)
        

 
def remove_users(tweet):
    return re.sub(r"@\S+", "", tweet)
    
   
  
def strip_whitespace(tweet):
    return ' '.join(tweet.split())



def rm_whitespace_before_punct(tweet):
    return re.sub(r'\s([?!"](?:\s|$))', r'\1', tweet)


def remove_symbols(tweet):
    return tweet.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")


 
def tokenize(tweet):
    return tokens_re.findall(tweet)
 
 
 
def preprocess(tweet, lowercase=False):
    tokens = tokenize(tweet)
    tokens = remove_stopwords(tokens)
    tokens = [token if emoticon_re.search(token) else token for token in tokens]
    tmp = remove_punctuation(tokens)
    return " ".join(tmp)



def remove_multiple_chars(tweet):
    return re.sub(r'(.)\1+', r'\1\1', tweet)



def remove_punctuation(tweet):
    regex_string = r'[^A-Za-z0-9!?]+'
    
    for i,token in enumerate(tweet):
        if len(token) < 2:
            tweet[i] = re.sub(regex_string, "", token)
    
    return tweet



def remove_digits(tweet):
    return ''.join([i for i in tweet if not i.isdigit()])

def remove_stopwords(tweet):
    return [token for token in tweet if token not in stop]

   
   
def remove_leftwhite(tweet):
    tweet = re.sub(" !", "!", tweet)
    tweet = re.sub(" \?", "?", tweet)
    return tweet



def normalize_tweet(tweet):
    tweet = remove_url(tweet)
    tweet = remove_users(tweet)
    tweet = remove_symbols(tweet).lower()
    tweet = remove_multiple_chars(tweet)
    #tweet = remove_digits(tweet)
    tweet = preprocess(tweet)
    tweet = strip_whitespace(tweet)
    return tweet



def tokenize_document(document):
    fields = ["id", "topic", "yLabel", "tweet"]
    dataframe = pd.read_table(document, header=None, names=fields)
    dataframe['tweet'] = dataframe['tweet'].apply(preprocess)
    dataframe.to_csv(r'../../data/normalized/twitter-2016test-outputNORMALIZED3.txt', header=None, index=None, sep='\t', mode='a')

    

def normalize_document(document):
    fields = ["id", "topic", "yLabel", "tweet"]
    dataframe = pd.read_table(document, sep="\t", header=None, names=fields)
    dataframe = dataframe[dataframe['tweet'] != "Not Available"]  
    dataframe['tweet'] = dataframe['tweet'].apply(remove_digits)
    dataframe.to_csv(r'../data/normalized/twitter-2016test-outputNORMALIZED2.txt', header=None, index=None, sep='\t', mode='a')

#normalize_document("../data/normalized/twitter-2016test-outputNORMALIZED.txt")   
tokenize_document("../../data/normalized/twitter-2016test-outputNORMALIZED2.txt")

print((normalize_tweet("@8dreamerMJJ thank god   23rd    ! i was. so nervous i didn't how? Michael, Jackson hardcore :) fans may feel about @chrisbrown taking on such a big project!!!")))

         
