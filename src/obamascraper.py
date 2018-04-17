#!/usr/bin/env python
# coding: utf8
from goose import Goose
import urllib
import lxml.html
import codecs

def get_links(url, domain):
    connection = urllib.urlopen(url)
    dom = lxml.html.fromstring(connection.read())
    for link in dom.xpath(‘//a/@href’): # select the url in href for all a tags(links)
    if ( link.startswith(“speech”) and link.endswith(“htm”) ):
    yield domain + link

def get_text(url):
    g = Goose() 
    article = g.extract(url=url)
    with codecs.open(article.link_hash + “.speech”, “w”, “utf-8-sig”) as text_file:
    text_file.write(article.cleaned_text)
    
if (__name__ == “__main__”):
    link = “http://www.americanrhetoric.com/barackobamaspeeches.htm"
    domain = “http://www.americanrhetoric.com/"
    for i in get_links(link, domain):
    get_text(i)