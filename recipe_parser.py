import nltk
from nltk.tag import pos_tag, map_tag

def get_features(text):
    words = []
    # Same steps to start as before
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = words + nltk.word_tokenize(sentence)

    # part of speech tag each of the words
    pos = pos_tag(words)
    # Sometimes it's helpful to simplify the tags NLTK returns by default.
    # I saw an increase in accuracy if I did this, but you may not
    # depending on the application.
    pos = [map_tag('en-ptb', 'universal', tag) for word, tag in pos]
    # Then, convert the words to lowercase like before
    words = [i.lower() for i in words]
    # Grab the trigrams
    trigrams = nltk.trigrams(words)
    # We need to concatinate the trigrams into a single string to process
    trigrams = ["%s/%s/%s" % (i[0], i[1], i[2]) for i in trigrams]
    # Get our final dict rolling
    features = words + pos + trigrams
    # get our feature dict rolling
    features = dict([(i, True) for i in features])
    return features

# Try it out
text = "Transfer the pan to a wire rack to cool for 15 minutes."
for key, value in get_features(text).items():
    print(key)