import os
import pickle
import gensim
from indexsystem import Indexsystem

class Indexed_Sentence:
    def __init__(self,indices,metadata):
        self.indices = indices
        self.metadata = metadata
        
class Filesystem:
    def __init__(self,rootdir,indexsystem,model,topvoc,voc_cutoff):
        
        self.lls = []
        for i in range(5,21):
            self.lls.append(LengthList(i,i+1,0,self))
        
        # self.lls.append(LengthList(40,50,0,self))
        # self.lls.append(LengthList(50,60,0,self))
        # self.lls.append(LengthList(60,70,0,self))
        # self.lls.append(LengthList(70,80,0,self))
        # self.lls.append(LengthList(80,100,0,self))
        
        self.mapping = {}
        index = 0
        for i in range(5,21):
            self.mapping[i]=index
            index += 1
        # for i in range(40,50):
        #     self.mapping[i] = index
        # index += 1
        # for i in range(50,60):
        #     self.mapping[i] = index
        # index += 1
        # for i in range(60,70):
        #     self.mapping[i] = index
        # index += 1
        # for i in range(70,80):
        #     self.mapping[i] = index
        # index += 1
        # for i in range(80,100):
        #     self.mapping[i] = index

        self.rootdir = rootdir
        if indexsystem:
            self.indexsystem = pickle.load(open(indexsystem,"rb"))
            self.save_indexsystem = False
        elif model or topvoc:
            self.indexsystem = Indexsystem(model,topvoc,voc_cutoff)
            self.save_indexsystem = True
        else:
            self.indexsystem = None
            self.save_indexsystem = False

    def finish(self):
        rest_ll = LengthList(5,0,0,self)
        for ll in self.lls:
            if len(rest_ll.data) + len(rest_ll.data) > 49:
                n_to_add = 50-len(rest_ll.data)
                rest_ll.data.extend(ll.data[:n_to_add])
                rest_ll.set_upper(ll.upper)
                rest_ll.write(self.rootdir)
                rest_ll = LengthList(ll.lower,ll.upper,0,self)
                rest_ll.data.extend(ll.data[n_to_add:])
            else:
                rest_ll.data.extend(ll.data)
        rest_ll.set_upper(99)
        rest_ll.write(self.rootdir)
        if self.save_indexsystem:
            self.indexsystem.write()
                
class LengthList:
    def __init__(self,lower,upper,index,fs):
        self.data = []
        self.lower = lower
        self.upper = upper
        self.index = index
        self.fs = fs

    def update(self,sentence,rootdir,metadata):
        if len(self.data) == 50:
            self.write(rootdir)
            self.data = [(sentence,metadata)]
            self.index += 1
        else:
            self.data.append((sentence,metadata))

    def convert_to_index(self,words):
        indices = []
        for word in words.split(" "):
            indices.append(self.fs.indexsystem.get_index(word))
        return indices

    def write(self,rootdir):
        filename = "sents_" + str(self.lower) + "_" + str(self.upper) + "_" + str(self.index)
        opf = open(os.path.join(rootdir,filename),"wb")
        #opf = open(os.path.join(rootdir,filename),"w")
        #pickle.dump
        sentences = []
        max_len = 0
        for (sentence,metadata) in self.data:
            sent = []
            for word in sentence.content:
                if isinstance(word,str):
                    final_word = word
                else:
                    final_word = word.text
                if self.fs.indexsystem:
                    sent.extend(self.convert_to_index(final_word))
                else:
                    sent_text = " ".join(sent)
                    sent = sent_text.split(" ")
            if len(sent) > max_len:
                max_len = len(sent)
            indsen = Indexed_Sentence(sent,metadata)
            sentences.append(indsen)
        #opf.write("\n".join(sentences))
        if max_len >= self.upper:
            print("ERROR : upper limit of %s, but sentence of length %s found." % (self.upper,max_len))
        pickle.dump(sentences,opf)
        opf.close()

    def set_lower(self,lower):
        self.lower = lower

    def set_upper(self,upper):
        self.upper = upper
