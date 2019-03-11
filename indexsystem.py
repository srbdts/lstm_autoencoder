import gensim
import pickle

class Indexsystem:
    def __init__(self,model,topvoc=None,voc_cutoff = None):
        if model:
            self.m = gensim.models.Word2Vec.load(model)
            if voc_cutoff:
                voc = self.m.wv.index2word[:voc_cutoff]
            else:
                voc = self.m.wv.index2word
        else:
            tv = pickle.load(open(topvoc,"rb"))
            voc = [0 for i in range(len(tv))]
            for (index,word) in tv.items():
                voc[int(index)] = word
            self.m = topvoc
        self.index_to_word = ["NULL","SOS","EOS","OOV"] + voc
        self.word_to_index = {word:index for index,word in enumerate(self.index_to_word)}
        self.vocsize = len(self.word_to_index)

    def get_index(self,word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index["OOV"]

    def write(self,file):
        pickle.dump(self,open(file,"wb"))
