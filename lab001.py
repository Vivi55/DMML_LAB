import nltk
import urllib.request
import nltk.tokenize as tk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
def meth1():
    sent = "I saw a boy in the park with a telescope."
    taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
    #print(taggedS)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(taggedS)
    print(result)
    result.draw()
#Who has the telescope? 'I' or 'boy'?

def meth2():
    grammar=nltk.parse()
    sent = "My life is brilliant. I saw an angle of that I'm sure. She smiled at me on the subway.".split()
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for p in rd_parser.nbest_parse(sent):
        print(p)

def test():
    with open("test.txt","r") as f1:
        doc = f1.read()
        tokens = tk.word_tokenize(doc)
        noun=[]
        #print(type(tokens)) # tokenize the words
        #result=all([words in tokens for words in the])


    for words in tokens:
        if bool(words,tokens):
            print(noun.append(words[0]))



        #    tokens[words] =='the'&'The':
         #       print(tokens[words+1])




              #  new_n = words.append(li)
                #print(new_n)
          #  else:
                #print('')
def bool(word_pos: list, wordList: list):
    if word_pos[0] in ["NN","NNS"]:
        po_num= wordList.index(word_pos)
        if po_num!=0 & wordList[po_num-1][0] in ["the", "The"]:
            return True
    return False


if __name__ == '__main__':
    test()







