
Dependencies
------------

Make sure the following python libraries are installed: keras, numpy, gensim


Running the training script
---------------------------

python train_lstm.py batch_files resources/embeddings resources/indexsystem <NAME_OF_OUTPUTMODEL> (--basemodel <NAME_OF_BASEMODEL>)

Each file in batch_files contains a pickled list of 50 indexed_sentence objects. Every object has an attribute "indices", where the actual indices of the sentence are stored.

The embeddings file contains word embeddings trained on a spelling-normalised 80-million word corpus with texts written between 1580 and 1600.

The indexsystem has two attributes: "word_to_index" is a dictionary that maps a word to its index; "index_to_word" is a list that maps an index to the word it represents. The indexsystem has been used to encode corpus sentences as lists of indices (such as the ones in the batch files) and to decode them into the original sentences again. The first four indices are preserved for special characters: 0 is mapped to "NULL" (padding), 1 is mapped to "SOS" (start of sequence), 2 is mapped to "EOS" (end of sequence) and 3 is mapped to "OOV" (out of vocabulary).

<NAME_OF_OUTPUTMODEL> is where the model will be saved after training. If this file already exists, it will be overwritten by the new model. In addition to the trained model, the script will also output a text file called "<NAME_OF_OUTPUTMODEL>.split", which contains information on which files have been used for training and which for testing.

--basemodel is an optional parameter to be set if you don't want to generate a model from scratch, but continue the training of an existing model (with new data). Useful if you want to initialize a model with the parameter settings of another one to ensure comparability.

Running the prediction script
-----------------------------

python predict_batch.py batch_files/<NAME_OF_INPUTFILE> <NAME_OF_MODEL> resources/indexsystem (--insert_correct True)

The first argument points to the batch file with indexed sentences that you want the model to encode and decode again.

The second argument points to the trained model that performs the prediction (i.e. the output of train_lstm.py)

The third argument points to the indexsystem to map indiced sentences to their originals (and vice versa)

The fourth argument is optional. If --insert_correct is set to True, the model will predict the next word from the ACTUAL previous word, rather than the word it just predicted. For example, assume that the first word was "because" but the model thought it was "but". If "--insert_correct" is set to True, the model will predict the next word based on the correct target sequence "START_OF_SEQUENCE because". If not set (the default option), the model will predict the next word based on the predicted target sequence "START_OF_SEQUENCE but". If not set, the errors the model makes early on will propagate to the rest of the prediction. If set, the prediction of the model might make little sense. As every word is computed based on the actual target rather than the predicted target, the prediction shouldn't be read as a sequence. If it contains three times "is" in a row, it doesn't mean that the model thinks 'is' is likely to follow 'is'. It means that it thought three times in a row that the next word in the actual sequence was 'is'.

The script will output the actual sentences and the model's predictions to the terminal. Save it to a file by adding "> OUTPUTFILE" to the command.



