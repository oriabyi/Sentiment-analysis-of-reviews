__author__ = 'Oleksandr Riabyi'

from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import Length, DataRequired

from flask import Flask, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap

from sentimentClassifier import SentimentClassifier
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
bootstrap = Bootstrap(app)

class NameForm(FlaskForm):
	def __init__(self):
		super(NameForm, self).__init__(csrf_enabled=False)
	name = TextAreaField("Review", validators=[DataRequired(), Length(1, 1000)])
	submit = SubmitField('Evaluate the tone of the review')

@app.route("/", methods=['GET', 'POST'])
def index():
	result = ''
	form = NameForm()
	if form.validate_on_submit():
		text = form.name.data
		classifier = SentimentClassifier()
		result = str(SentimentClassifier().analysis_review(text))
	return render_template('index.html', form=form, name=result)

if __name__ == "__main__":
	app.run(port=5050)
