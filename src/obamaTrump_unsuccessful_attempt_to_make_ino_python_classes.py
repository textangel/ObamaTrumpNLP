from sklearn.naive_bayes import MultinomialNB
from nltk import sent_tokenize, word_tokenize, pos_tag
import codecs, sys, re, pdb, random, numpy as np, nltk
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize.api import StringTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid

class ObamaTrumpModel:
    def __init__(self):
        print "start init"
        self.__main__()
        pass


    def getCleanedData(path):
        data = []
        with open(path,"r") as f:
            data = f.read()
        udata = data.decode("utf-8")
        data = udata.encode("ascii","ignore")
        ansi_escape = re.compile(r'/(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]/')
        data = ansi_escape.sub('', data)
        whitespace_escape = re.compile(r'[\r\n\t]')
        data = whitespace_escape.sub('', data)
        quotes_escape = re.compile("\\'")
        data= whitespace_escape.sub("'", data)
        data = data.replace(" --",",")
        sentense_end_regex = re.compile("\.([a-zA-Z])")
        data= sentense_end_regex.sub(r'. \1', data)
        return data
    #     print data
        # sent_text = nltk.sent_tokenize(data)

    def tokenizeAndFilter(data_sent):
        data_sent2 = []
        ld = len(data_sent)
        cnt = 0
        for sent in data_sent:
            cnt+=1
            print ('Processed %d out of %d' % (cnt,ld))
            words = [w.lower() for w in nltk.word_tokenize(sent) if w.isalpha() and w.lower() not in stopwords.words("english")]
            if (len(words) > 4):
                data_sent2 = data_sent2 + [' '.join(words)]
        return data_sent2


    def simpleFilter(data_sent):
        data_sent2 = []
        ld = len(data_sent)
        cnt = 0
        for sent in data_sent:
            cnt+=1
            print ('Processed %d out of %d' % (cnt,ld))
            words = [w for w in nltk.word_tokenize(sent) if w.isalpha() and w.lower() not in stopwords.words("english")]
            if (len(words) > 4):
                data_sent2 = data_sent2 + [sent]
        return data_sent2

    def getData(self):
        obama = self.getCleanedData("../docs/allobamaspeeches.txt")
        trump = self.getCleanedData("../docs/trumpspeeches.txt")
        obama_sent = nltk.sent_tokenize(obama)
        trump_sent = nltk.sent_tokenize(trump)
        return obama_sent, trump_sent

    def getModel(tr,y):
        parameters = {'vect__ngram_range': [(1, 2),(1,3)],
                      'tfidf__use_idf': [False],
                      'vect__stop_words': [None],
                      'vect__binary':[True, False],
                      'clf__loss': [ 'modified_huber', 'perceptron']
                      }


        text_clf = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', SGDClassifier(penalty='l2',
                                                    alpha=1e-3, random_state=23))])
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, scoring = 'f1')
        gs_clf.fit(tr,y)
        return gs_clf

    def printGsResults(gs_clf):
        print("Best parameters set found on development set:\n")
        print(gs_clf.best_params_)
        print("Grid scores on development set:\n")
        means = gs_clf.cv_results_['mean_test_score']
        stds = gs_clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
   
    def printTestResults(gs_clf, te_y, test):
        print("Detailed classification report:\n")
        print("The model is trained on the full development set.\n")
        print("The scores are computed on the full evaluation set.\n")
        y_true, y_pred = te_y, gs_clf.predict(test)
        print(classification_report(y_true, y_pred))
        print("\n")

    def testOnOutsideData(gs_clf, self):
        test = self.getCleanedData("./tests/test_doc.txt")
        test_sent = nltk.sent_tokenize(test)
        tst = simpleFilter(test_sent)
        print gs_clf.predict(tst)

    def __main__(self):
        print "start main"
            # data_sent2 = [s for s in data_sent if len(s) > 20] #char length not word length
        o,t = self.getData()
        # o_old2, t_old2 = list(o), list(t)
        o = simpleFilter(o)
        t = simpleFilter(t)
        o = random.sample(o, len(t))
        #making training sets
        o_tr, o_te = train_test_split(o, test_size = 0.2)
        t_tr, t_te = train_test_split(t, test_size = 0.2)
        o_y = [1 for x in range(len(o_tr))]
        t_y = [0 for x in range(len(t_tr))]
        tr = o_tr + t_tr
        y = o_y + t_y
        #making test sets
        test = o_te + t_te
        ote_y = [1 for x in range(len(o_te))]
        tte_y = [0 for x in range(len(t_te))]
        te_y = ote_y + tte_y

        gs_clf = getModel(tr,y)
        printGsResults(gs_clf)
        printTestResults(gs_clf, te_y, test)
        testOnOutsideData(gs_clf, self)

ObamaTrumpModel()

    # if gs_clf.predict(tst)[0] == 0: print "Trump"
    # else: print "Obama"
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                       ('tfidf', TfidfTransformer()),
    #                       ('naive_bayes', MultinomialNB())])

