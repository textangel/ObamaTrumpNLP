# ObamaTrumpNLP
See the Obama - Trump NLP Sentense Generator.ipynb file for the project

Uses sklearn and an n-gram model. Reads a corpus of Obama's speeches and Trump's speeches and cleans them. Then splits them into sentences and uses them (all of Trump's sentences and a random subset of Obama's sentences of the same size) to train an n-gram based SVM classifier with parameters tuned with GridSearch. Tests the model on data, achieving 90 percent precision and recall (2017.8.25). We then take quotes from famous people and compute whether they speak more like Obama or more like Trump, on a sentence-by-sentence basis.
