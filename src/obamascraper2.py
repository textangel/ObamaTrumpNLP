from bs4 import BeautifulSoup, NavigableString
import urllib2,sys,datetime,json,re, codecs
from StdSuites.Type_Names_Suite import null
from docutils.nodes import paragraph

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8'}


def getHTMLSoupFromAddress(address):
    # address = "http://www.staradvertiser.com/2017/08/08/breaking-news/u-s-scientists-contradict-trumps-climate-claims/"
    req = urllib2.Request(address, headers=hdr)
    try:
        page = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        print e.fp.read()    
    html = page.read()
#     print html
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def clean(text):
    text = re.sub(u"(\u2018|\u2019)", "'", text)
    text = re.sub(u"(\u201c|\u201d)", '"', text)
    return text

def getDataFromAddress(address = "http://www.staradvertiser.com/2017/08/08/breaking-news/u-s-scientists-contradict-trumps-climate-claims/"):
    soup = getHTMLSoupFromAddress(address)

    paragraphs = ""
    for p in soup.find_all(size = "2", face = "Verdana"):
        str = p.string
        if str != None and str != "":
            str = str.rstrip()
            paragraphs = paragraphs + " " + str
            
    paragraphs = clean(paragraphs)
    print paragraphs
    return paragraphs


def getAllUrls(address= "http://www.americanrhetoric.com/barackobamaspeeches.htm"):
    soup = getHTMLSoupFromAddress(address)
    urls = []
    for link in soup.find_all(href=re.compile("^/?speeches/[^(PDF)]")):
        link = 'http://www.americanrhetoric.com/' + link.get('href')
        link = link.encode('ascii', 'ignore')
        urls = urls + [link]
#     print urls
    return urls



def getAllData(url):    
    urls = getAllUrls("http://www.americanrhetoric.com/barackobamaspeeches.htm")
#     print urls
    try:
        file = codecs.open("allobamaspeeches2.txt","w+", encoding="ascii") 
        for url in urls:
            print "Writing file" + url
            articleData = getDataFromAddress(url)
#             print articleData
            file.write(articleData)
            file.write("\n\n")
            print "finished writing file" + url
        file.close() 
    except:
        print "Unexpected error:", sys.exc_info()[0]

# getDataFromAddress("http://www.americanrhetoric.com/speeches/convention2004/barackobama2004dnc.htm")
getAllData("http://www.americanrhetoric.com/barackobamaspeeches.htm")
# getAllUrls()
