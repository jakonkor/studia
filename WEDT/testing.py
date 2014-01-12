#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import MySQLdb
import re
from bs4 import BeautifulSoup


from sklearn.datasets import load_files
from sklearn.feature_extraction import text
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import svm

page = """
<HTML>
<HEAD>
<TITLE>Your Title Here</TITLE>
</HEAD>
<BODY BGCOLOR="FFFFFF">
<CENTER><IMG SRC="clouds.jpg" ALIGN="BOTTOM"> </CENTER>
<HR>
<div>
<a href="http://somegreatsite.com">Link Name</a>
is a link to another nifty site
<div><H1>This is a Header</H1></div><div><H1>This is a Header</H1></div><div><H1>This is a Header</H1></div>
<div>
<div class="article-entry text">
<H2>This is a Medium Header</H2>
Send me mail at <a href="mailto:support@yourcompany.com">
support@yourcompany.com</a>.
<P> This is a new paragraph!
<P> <B>This is a new paragraph!</B>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<HR>
</div>
</div>
<div class="article-entry text">
<H2>This is a Medium Header 2 </H2>
Send me mail at <a href="mailto:support@yourcompany.com">
support@yourcompany.com</a>.
<P> This is a new paragraph!
<P> <B>This is a new paragraph!</B>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<HR>
</div>
</div>
</BODY>
</HTML>
"""

soup = BeautifulSoup(page)
corpus_post = []
y_labels = []
#divs = soup.find_all('div')
divs = soup.div.find_all(True, recursive=False)
content = BeautifulSoup(str(soup.find_all(attrs={"class": "article-entry text"})))
tags = soup.div.find_all(True)
post_divs = soup.find_all(attrs={"class": "article-entry text"})
print(len(post_divs))
for tag in tags :
    print('<' * 80)
    #for child in div.descendants :
        #if print(child)
    #print(div)
    #if div.attrs :
     #         if attr.conta != None :
       #         if attr.value == 'article-entry text' :
      #              print(attr.value)
    if tag in post_divs :
        print("ARTYKUŁ")
        corpus_post.append(str(tag))
        y_labels.append('post')
    else:
        print(tag)
        corpus_post.append(str(tag))
        y_labels.append('other')
    print('>' * 80)
    
print('_' * 80)
print('_' * 80)
#for string in soup.stripped_strings:
#    print(string)

#content = BeautifulSoup(str(soup.find_all(attrs={"class": "article-entry text"})))

#for string in content.stripped_strings:
#    print(string)

#<div class="article-entry text">  techcrunch
# <div class="news-content">  antyweb
# <div xmlns:fo="http://www.w3.org/1999/XSL/Format" class="cmsArtykulElem">  gazeta 


#-------------------------------------
# wektoryzacja 

print(corpus_post)
bigram_vectorizer = text.CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)

x_train = bigram_vectorizer.fit_transform(corpus_post)

print(x_train.toarray())

analyzer = bigram_vectorizer.build_analyzer()


clf = MultinomialNB(alpha=.01)
clf.fit(x_train,y_labels)
clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(x_train)

test = """
<H2>This is a Medium Header 2 </H2>
Send me mail at <a href="mailto:support@yourcompany.com">
support@yourcompany.com</a>.
<P> This is a new paragraph! 
fadfhalflkadjfa
<P> <B></B>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<H2>This is a Medium Header 2 </H2>
Send me mail at <a href="mailto:support@yourcompany.com">
support@yourcompany.com</a>.
<P> This is a new paragraph!
<P> <B>This is a new paragraph!</B>

fafafafdalfajkf
dfs

sf
ds
fs
f
sf
sd
fs
fsd
f
s
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
<H2>This is a Medium Header 2 </H2>
Send me mail at <a href="mailto:support@yourcompany.com">
support@yourcompany.com</a>.
<P> This is a new paragraph!
<P> <B>This is a new paragraph!</B>
<div><BR> <B><I>This is a new sentence without a paragraph break, in bold italics.</I></B></div>
"""
X_test = []
X_test.append(test)
X_test.append('to nie jest artykul raczej')
X_test.append('<div><H1>gsdgdsjgs.jdhgfjdsgjsdhglksf;dgsdgld</H1></div>')
pred = clf.predict(bigram_vectorizer.transform(X_test))
print(pred)
pred1 = clf.predict(x_train)
print(pred1)
