from __future__ import division
import nltk
from nltk.probability import FreqDist
import io

f = io.open('mobydick.txt', 'rU', encoding='utf-8')
raw = f.read()
tokens = nltk.word_tokenize(raw)

print ('tokens: ', len(tokens))

def get_meaninful_content(text):
  stopwords = nltk.corpus.stopwords.words('spanish')
  #content = [w for w in text if w.lower() not in stopwords]
  content = [w for w in text if w.lower() not in stopwords and len(w)>1]
  return content

content = get_meaninful_content(tokens)

print ('content: ', len(content))
#print fdist.N()

def content_fraction(text, content):
  return len(content) / len(text)

print ('content_fraction: ', content_fraction(tokens, content))

text = nltk.Text(content)

print ('unique tokens: ', len(set(text)))

richness = len(text)/len(set(text))
print ('richness: ', richness)

#print sorted(set(text))

fdist = FreqDist(content)

#vocab = fdist.keys()
#print vocab[:25]

#fdist.plot(50, cumulative=True)

#mostFreqW = sorted([w for w in set(text) if fdist[w]>6])

#for w in mostFreqW:
#  print w.encode('utf-8') , ': ' , fdist[w]

#print fdist.most_common(n=5)
mostFreqWDist = fdist.most_common(20)
#print mostFreqWDist
for t in mostFreqWDist:
  print (t[0].encode('utf-8'), ':', t[1])

mostFreqW = [ t[0] for t in mostFreqWDist ]

def get_full_text(text):
  #content = [w for w in text if w.lower() not in stopwords]
  full_content = [w for w in text]
  full_text = nltk.Text(full_content)
  return full_text

full_text = get_full_text(tokens)

#full_text.concordance(u'pap\xe1')

#full_text.dispersion_plot(mostFreqW)

'''
import matplotlib.pyplot as plt
x, y = zip(*fdist.most_common(n=5s))
plt.bar(range(len(x)), y)
plt.xticks(range(len(x)), x)
plt.show()
'''
