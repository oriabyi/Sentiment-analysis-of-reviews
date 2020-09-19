import dill
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

with open('./pickles/vectorizer.pickle',  'rb') as f:
	vectorizer  = dill.load(f)

with open('./pickles/model.pickle',       'rb') as f:
	model       = dill.load(f)
	
if __name__ == '__main__':
	with open('./data/test.csv', 'r') as f:
	    df = f.readlines()
	    df_cleaned = '\n'.join(df).replace('<review>', '').replace('\n', ' ').split('</review>')
	    df_cl_stripped = [el.strip() for el in df_cleaned]
	    test = [el for el in df_cl_stripped if el]

	print(model.predict(vectorizer.transform(test)))
