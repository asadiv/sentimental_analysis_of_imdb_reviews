import pyprind
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


# this implementation is usefull when you have large amounts fo data or it is continously flowing in
# it would be too hard to build the vocabulary dictionary as the whole data wont fitin in memeory
# also here you cant use countvectorizer and tf-idf as they req full data, so we would be using hashingvectorizor


# along with tokeniztion we also added some cleaning steps
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# it gets the next review and label from file
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# gets a batch acc to size param
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

#  uses murmurhash function, greater number such as 2**21 for n_features avoids collision
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

# this model supports partialfit whcih is necessary for batch wise training
# if we want to use gridsearch then we would have to setup our own as the one in scikit learn assumes that whole data is provided at once
# we could simply setup our own whcih will carry out the training in diff way ie in chunks
clf = SGDClassifier(loss='log', random_state=1)

doc_stream = stream_docs(path='movie_data.csv')

pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')

# fit on remaing data
clf = clf.partial_fit(X_test, y_test)