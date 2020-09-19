import dill
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentClassifier(object):
	def __init__(self):
		with open('./pickles/vectorizer.pickle',  'rb') as f:
			self.vectorizer  = dill.load(f)

		with open('./pickles/model.pickle',       'rb') as f:
			self.model       = dill.load(f)

	def analysis_review(self, review:str, pretty_answer:bool=True) -> list:
		if pretty_answer:
			return self.pretty_answer(self.model.predict_proba(self.vectorizer.transform([review]))[0])

	def pretty_answer(self, probs:list) -> str:
		if probs[1] >= .5:
			return f"This is a {int(round(probs[1] * 100)) }% percent POSITIVE review" 
		else:
			return f"This is a {int(round(probs[0] * 100)) }% percent NEGATIVE review"


if __name__ == '__main__':
	test = SentimentClassifier()
	review = "so , why the small digital elph , rather than one of the other cameras with better resolution or picture quality ? size [ + 2 ] # # because , unless it 's small , i won 't cary it around ."
	print(test.analysis_review(review))

