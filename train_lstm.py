from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import *
import numpy as np
import gensim
import os, pickle, sys, random, argparse
from indexsystem import *
from filesystem import *
from keras.models import load_model
import timeit

parser = argparse.ArgumentParser()
parser.add_argument("rootdir")
parser.add_argument("embeddingsmodel")
parser.add_argument("indexsystem")
parser.add_argument("outputmodel")
parser.add_argument("--basemodel",dest="basemodel",default=False)
args = parser.parse_args()

# Set hyperparameters
BATCH_SIZE = 50
EPOCHS = 1
LATENT_DIM = 100
EMBEDDING_DIM = 400
VOC_CUTOFF = 10000
MAX_LEN = 20
SPLITFILE = args.outputmodel + ".split"
ROOTDIR = args.rootdir
MODEL = args.outputmodel
EMBEDDINGS_PATH = args.embeddingsmodel
INDEXSYSTEM_PATH = args.indexsystem
BATCH_FILES = os.listdir(ROOTDIR)
TEST_PROP = 0.2

#startseq = np.random.randn(EMBEDDING_DIM)
#endseq = np.random.randn(EMBEDDING_DIM)
#oov = np.random.randn(EMBEDDING_DIM)
# np.save(open("startseq.npy","wb"),startseq)
# np.save(open("endseq.npy","wb"),endseq)
# np.save(open("oov.npy","wb"),oov)

#class TimeTraining(Callback):
#    def on_batch_begin(self,batch,logs={}):
#        self.starttime = timeit.default_timer()
#    def on_batch_end(self,batch,logs={}):
#        self.endtime = timeit.default_timer()
#        print("\nbatch execution time: %s" % (self.endtime-self.starttime))

# Load special vectors to ensure repeateability
startseq = np.load("resources/startseq.npy")
endseq = np.load("resources/endseq.npy")
oov = np.load("resources/oov.npy")


# Load embeddingsmodel and generate embeddingmatrix
indexsystem = pickle.load(open(INDEXSYSTEM_PATH,"rb"))
m = gensim.models.Word2Vec.load(EMBEDDINGS_PATH)
index_to_word = indexsystem.index_to_word
word_to_index = indexsystem.word_to_index 
VOCSIZE = indexsystem.vocsize
embedding_matrix = np.zeros((VOCSIZE,EMBEDDING_DIM))
for i in range(0,len(index_to_word)):
    try:
        embedding_matrix[i] = m.wv[index_to_word[i]]
    except KeyError:
        embedding_matrix[i] = oov

embedding_matrix[word_to_index["SOS"]] = startseq
embedding_matrix[word_to_index["EOS"]] = endseq
embedding_matrix[word_to_index["OOV"]] = oov

# Split data in test and train set
TRAIN_FILES = []
TEST_FILES = []
indices = [i for i in range(0,len(BATCH_FILES))]
random.shuffle(indices)
test_size = round(len(BATCH_FILES)*TEST_PROP)
test_indices = indices[:test_size]
train_indices = indices[test_size:]
TEST_FILES = [BATCH_FILES[i] for i in test_indices]
TRAIN_FILES = [BATCH_FILES[i] for i in train_indices]
print("Training on %s files" % len(TRAIN_FILES))
print("Testing on %s files" % len(TEST_FILES))
opf = open(SPLITFILE,"w")
opf.write("##TRAIN##\n%s\n##TEST##\n%s\n" % ("\n".join(TRAIN_FILES),"\n".join(TEST_FILES)))

N_BATCHES = len(TRAIN_FILES)

# Load input from batch files and configure properly for model
def myGenerator(batch_files):
    while True:
        for bf in batch_files:
            upper = int(bf.split("_")[-2]) # last number of batch filename indicates maximal length of sentences in that file
            # Initialise input/output vectors
            encoder_input_data = np.zeros((BATCH_SIZE,upper+1),dtype="float32")
            decoder_input_data = np.zeros((BATCH_SIZE,upper+2),dtype="float32")
            decoder_output_data = np.zeros((BATCH_SIZE,upper+2),dtype="float32")
            #load sentences
            indexed_sentences = pickle.load(open(os.path.join(ROOTDIR,bf),"rb"))

            if len(indexed_sentences) > BATCH_SIZE:
                print("file %s contains too many sentences" % (bf))
                indexed_sentences = indexed_sentences[:50]
            for i,s in enumerate(indexed_sentences):
                s = s.indices
                encoder_input_data[i,:len(s)] = s
                decoder_input_data[i,0] = word_to_index["SOS"]
                decoder_input_data[i,1:len(s)+1] = s
                decoder_output_data[i,:len(s)] = [index if index < 10000 else word_to_index["OOV"] for index in s]
                decoder_output_data[i,len(s)] = word_to_index["EOS"]
            # Convert decoder output to one-hot encoding
            decoder_matrix = np.zeros((len(decoder_output_data),upper+2,VOC_CUTOFF))
            for i,mat in enumerate(decoder_output_data):
                decoder_matrix[i] = to_categorical(mat,VOC_CUTOFF)
            yield [encoder_input_data,decoder_input_data],decoder_matrix

# If a model was trained on the previous time slice, take that model and continue training
if args.basemodel:
    model = load_model(args.basemodel)
    print(model.summary())
    model.fit_generator(myGenerator(TRAIN_FILES),steps_per_epoch=N_BATCHES,epochs=EPOCHS)
    model.save(MODEL)
# If not, generate model from scratch
else:
    # convert encoder inputs to embeddings
    encoder_input_raw = Input(shape=(None,))
    embedding_layer_encoder = Embedding(VOCSIZE,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False,input_length=None,mask_zero=True)
    encoder_inputs = embedding_layer_encoder(encoder_input_raw)
    # create encoder LSTM layer
    encoder = LSTM(LATENT_DIM,return_state=True)
    # feed encoder input embeddings to encoder LSTM and keep track of hidden states and cell states
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h,state_c]

    # convert decoder inputs to embeddings
    decoder_input_raw = Input(shape=(None,))
    embedding_layer_decoder= Embedding(VOCSIZE,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False,input_length=None,mask_zero=True)
    decoder_inputs = embedding_layer_decoder(decoder_input_raw)
    # create decoder LSTM layer
    decoder = LSTM(LATENT_DIM,return_sequences=True,return_state=True)
    # feed decoder input embeddings to decoder LSTM; initialise with final hidden states of encoder
    decoder_outputs, _, _ = decoder(decoder_inputs,initial_state=encoder_states)
    # softmax layer to map decoder outputs to output classes
    decoder_dense = Dense(VOC_CUTOFF,activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_input_raw,decoder_input_raw],decoder_outputs)
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    #cb = TimeTraining()
    #print(cb)
    model.fit_generator(myGenerator(TRAIN_FILES),steps_per_epoch=N_BATCHES,epochs=EPOCHS)
    model.save(MODEL)
