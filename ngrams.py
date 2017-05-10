from __future__ import division
import nltk
from nltk.collocations import *
#from nltk.probability import FreqDist
import io

#from nltk import bigrams, trigrams

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

f = io.open('mobydick.txt', 'rU', encoding='utf-8')
raw = f.read()
#tokens = nltk.word_tokenize(raw)
tokens = nltk.wordpunct_tokenize(raw)

def get_meaninful_content(text):
  stopwords = nltk.corpus.stopwords.words('reddit')
  #content = [w for w in text if w.lower() not in stopwords]
  content = [w for w in text if w.lower() not in stopwords and len(w)>1]
  return content

content = get_meaninful_content(tokens)

def get_full_text(text):
  #content = [w for w in text if w.lower() not in stopwords]
  full_content = [w for w in text]
  full_text = nltk.Text(full_content)
  return full_text

full_text = get_full_text(tokens)


#full_text.concordance('alianza')
full_text.collocations()

content_text = nltk.Text(content)
content_text.collocations()

## Bigrams
finder = BigramCollocationFinder.from_words(content)
 
# only bigrams that appear 5+ times
finder.apply_freq_filter(2)
scored = finder.score_ngrams(bigram_measures.raw_freq)
bigrams_sorted = sorted(bigram for bigram, score in scored) 

for bigram in bigrams_sorted:
  print (bigram[0].encode('utf-8'), bigram[1].encode('utf-8'))


## Trigrams
finder = TrigramCollocationFinder.from_words(content)
 
# only bigrams that appear 5+ times
finder.apply_freq_filter(2)
scored = finder.score_ngrams(trigram_measures.raw_freq)
trigrams_sorted = sorted(trigram for trigram, score in scored) 

for trigram in trigrams_sorted:
  print (trigram[0].encode('utf-8'), trigram[1].encode('utf-8'), trigram[2].encode('utf-8'))

''' if it is traigram don't consider it's componets as bigrams '''