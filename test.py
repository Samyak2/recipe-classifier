import nltk
import pickle
from nltk.classify import MaxentClassifier
from nltk.tag import pos_tag, map_tag

# Set up our training material in a nice dictionary.
training = {
    'ingredients': [
        'Pastry for 9-inch tart pan',
        'Apple cider vinegar',
        '3 eggs',
        '1/4 cup sugar',
    ],
    'steps': [
        'Sift the powdered sugar and cocoa powder together.',
        'Coarsely crush the peppercorns using a mortar and pestle.',
        'While the vegetables are cooking, scrub the pig ears clean and cut away any knobby bits of cartilage so they will lie flat.',
        'Heat the oven to 375 degrees.',
    ]
}

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

    bigrams = nltk.bigrams(words)
    bigrams = ["%s/%s" % (i[0], i[1]) for i in bigrams]

    # Get our final dict rolling
    features = words + trigrams + bigrams
    # get our feature dict rolling
    features = dict([(i, True) for i in features])
    return features

# Set up a list that will contain all of our tagged examples,
# which we will pass into the classifier at the end.
training_set = []
for key, val in training.items():
    for i in val:
        # Set up a list we can use for all of our features,
        # which are just individual words in this case.
        feats = get_features(i)
        # Before we can tokenize words, we need to break the
        # text out into sentences.
        # sentences = nltk.sent_tokenize(i)
        # for sentence in sentences:
            # feats = feats + nltk.word_tokenize(sentence)
        # For this example, it's a good idea to normalize for case.
        # You may or may not need to do this.
        # feats = [i.lower() for i in feats]
        # Each feature needs a value. A typical use for a case like this
        # is to use True or 1, though you can use almost any value for
        # a more complicated application or analysis.
        # feats = dict([(i, True) for i in feats])
        # NLTK expects you to feed a classifier a list of tuples
        # where each tuple is (features, tag).
        training_set.append((feats, key))
for value in training_set:
    # for data, key in value:
    print(value[1])
    print(value[0])
    print("-----")

# Train up our classifier
classifier = MaxentClassifier.train(training_set)

# Test it out!
# You need to feed the classifier your data in the same format you used
# to train it, in this case individual lowercase words.
print(classifier.classify({'apple': True, 'cider': True, 'vinegar': True}))

outfile = open('my_pickle.pickle', 'wb')
pickle.dump(classifier, outfile)
outfile.close()