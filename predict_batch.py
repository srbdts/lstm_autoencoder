
from keras.models import Model
import numpy as np
import os,pickle,sys,argparse
from keras.models import load_model
from indexsystem import *
'''
    Next: inference mode (sampling):
    Here's the drill:
    1) encode input and retrieve initial decoder state
    2) run one step of decoder with this initial state and a 'start of sequence token' as target. Output will be the next token
    3) Repeat with the current target token and current states
    
'''

parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("model")
parser.add_argument("indexsystem")
parser.add_argument("--insert_correct",dest="insert_correct",default=None)
args = parser.parse_args()


# Define sampling models

model = load_model(args.model)
encoder_inputs = model.get_layer("input_1").input
decoder_inputs = model.get_layer("input_2").input
decoder_outputs = model.get_layer("dense_1").output
decoder_model = Model([encoder_inputs,decoder_inputs],decoder_outputs)

def autoencode_sequence(input_seq):
    predictions = decoder_model.predict(input_seq)
    return predictions


insy = pickle.load(open(args.indexsystem,"rb"))

testdata = pickle.load(open(args.inputfile,"rb"))
upper = int(args.inputfile.split("_")[-2])
max_loops = 50
encoder_input_data = np.zeros((len(testdata),upper+1),dtype="float32")
if args.insert_correct:
    decoder_input_data = np.zeros((len(testdata),upper+2),dtype="float32")
else:
    # One is the index of the start-of-sequence sign.
    decoder_input_data = np.ones((len(testdata),max_loops),dtype="float32")

for i,s in enumerate(testdata):
    encoder_input_data[i,:len(s.indices)] = s.indices
    if args.insert_correct:
        decoder_input_data[i,0] = 1
        decoder_input_data[i,1:len(s.indices)+1] = s.indices


predictions = autoencode_sequence([encoder_input_data,decoder_input_data])
if args.insert_correct:
    predicted_indices = []
    for s in range(len(predictions)):
        predicted_indices.append([])
        for w in range(len(predictions[s])):
            predicted_indices[-1].append(np.argmax(predictions[s,w]))
else:
    predicted_indices = [[] for i in range(len(testdata))]
    finished = set()
    loop = 0
    while len(finished) < len(testdata) and loop < max_loops:
        for s in range(len(predictions)):
            #print("actual sentence   : %s" % (" ".join([insy.index_to_word[i] for i in testdata[s].indices])))
            predicted_index = np.argmax(predictions[s,loop])
            predicted_indices[s].append(predicted_index)
            if predicted_index == 2:
                finished.add(s)
            decoder_input_data[s,loop+1] = predicted_index
        predictions = autoencode_sequence([encoder_input_data,decoder_input_data])
        loop += 1

for nr,predicted_sentence in enumerate(predicted_indices):
    print("actual sentence   : %s" % (" ".join([insy.index_to_word[i] for i in testdata[nr].indices])))
    print("predicted sentence: %s" % (" ".join([insy.index_to_word[i] for i in predicted_sentence])))
#decoded_sentence = decode_sequence(decode_input_data)
#print("Input sentence:", line)
#print("Decoded sentence:", decoded_sentence)


