#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import MySQLdb
import re
from bs4 import BeautifulSoup


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn import svm


def strip_tags(html, whitelist=[]):
    """
    Strip all HTML tags except for a list of whitelisted tags.
    """
    soup = BeautifulSoup(html)

    for tag in soup.findAll(True):
        if tag.name not in whitelist:
            tag.append(' ')
            tag.replaceWithChildren()

    result = unicode(soup)

    # Clean up any repeated spaces and spaces like this: '<a>test </a> '
    result = re.sub(' +', ' ', result)
    result = re.sub(r' (<[^>]*> )', r'\1', result)
    result = re.sub(r'\s', ' ', result)
    result = re.sub(r'(<!--.*?-->)' ,' ', result)
    result = re.sub(r'\\.' ,' ', result)
    result = re.sub(r'\(\'|\',\)' ,' ', result)
    result = re.sub(r'  ' ,' ', result)
    return result.strip()

    
db = MySQLdb.connect(host="serwer1363634.home.pl", # your host, usually localhost
                     user="13777141_wedt", # your username
                      passwd="wedt123", # your password
                      db="13777141_wedt") # name of the data base

# you must create a Cursor object. It will let
#  you execute all the query you need
cur = db.cursor() 

# Use all the SQL you like
cur.execute("SELECT content FROM parser_results")

text_test = """ 
<div class="article-entry text">

<!-- Begin: Wordpress Article Content -->
<img src="http://tctechcrunch2011.files.wordpress.com/2014/01/screen-shot-2014-01-01-at-9-04-42-am.png?w=738" class=""><p>According to <a target="_blank" href="http://www.netmarketshare.com/operating-system-market-share.aspx?qprid=8&amp;qpcustomd=0">Net Applications</a>, Windows 8.x crossed the 10% barrier in December of 2013. Windows 8 and Windows 8.1 ended the year with 6.89% and 3.60% apiece for a combined 10.49% total market share.</p>
<p>In the month, Windows 7 <a target="_blank" href="http://thenextweb.com/insider/2014/01/01/windows-8-windows-8-1-pass-10-market-share-windows-7-still-gains-windows-xp-falls-30/">picked up 0.88% market share</a>, as Windows XP fell below the 30 percent mark, shedding 2.24% to land at 28.98% in the month. While Windows 8.x’s market share growth is probably still under what Microsoft wants, enterprise adoption of Windows 7 appears strong as the end of Windows XP approaches.</p>
<p>Windows 8 gained 0.23% market share in December, an almost surprising figure given the general availability of Windows 8.1, a free upgrade. The latter did pick up 0.96% in the month.</p>
<p>It will be interesting to see how Windows 8.x’s growing market share converts into downloads of applications through the Windows Store. Previously, the Windows developer portal provided <a href="http://techcrunch.com/2013/12/03/paid-game-downloads-spike-34-on-the-windows-store-in-november-but-microsoft-obscures-other-data-points/">detailed download numbers</a>. However, this morning I was unable to load the usual set of analytics through the system. Microsoft may have removed the capability.</p>
<p>If so, we will not be able to correlate downloads with market share, which will limit our ability to vet Microsoft’s ability to convert new Windows 8.x users to its new application platform. That’s a shame.</p>
<p>To wrap 2013, Windows 8.x manages a new threshold as Windows 7 manages to accelerate the end of Windows XP. Not a bad way to start 2014.</p>
<p><em>Top Image Credit: <a target="_blank" href="http://www.flickr.com/photos/dellphotos/">Flickr</a></em></p>
<!-- End: Wordpress Article Content -->

								
							</div>
"""
i = 0
text = """ 

"""

print(strip_tags(str(text_test), []))
y_train = []
f = open('tmp.txt','w')
f.write('START\n')
#print strip_tags(text_test, [])
# print all the first cell of all the rows
for article in cur.fetchall() :
    #text = text + strip_tags(str(content), [])
    i = i + 1
    f = open('data/post/' + str(i) + '.txt','w')
    f.write(str(article) + '\n')
    f.write(strip_tags(str(article), []))
    f.close
 #   print text
  #  print(strip_tags(text, []), file="tmp.dat")
    f.write(strip_tags(text, []))
f.write('END\n')
f.close()    
print("data building: DONE!!!")

#dataset = load_files('data')
dataset = load_files('data_one_class')
y_train.append('post')
y_train.append('post')
y_train.append('post')
y_train.append('post')
y_train.append('post')
print("dataset building: DONE")
#print(dataset)
vectorizer = HashingVectorizer(non_negative=True)
#vectorizer = TfidfVectorizer()
#data_train = vectorizer.transform(dataset.data)
data_train = vectorizer.transform(dataset)
print("data_train building: DONE!!!")
print(data_train)


clf = MultinomialNB(alpha=.01)
clf.fit(data_train, y_train)
clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(data_train)

pred = clf2.predict(vectorizer.transform(['<div>to raczej nie post</div>']))
pred = clf2.predict(data_train)
print(pred)