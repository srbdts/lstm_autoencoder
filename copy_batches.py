import argparse, os, random
import pickle

p = argparse.ArgumentParser()
p.add_argument("inputdir")
p.add_argument("outputdir")
p.add_argument("cutoff")
args = p.parse_args()

args.cutoff = int(args.cutoff)


files = os.listdir(args.inputdir)
indices = random.sample(range(len(files)),args.cutoff)
for index in indices:
    sentences = pickle.load(open(os.path.join(args.inputdir,files[index]),"rb"))
    length_index = files[index].split("_")[-1]
    length = len(sentences[0].indices)-4
    new_batch = []
    for s in sentences:
        s.indices = s.indices[2:-2]
        new_batch.append(s)
    new_file = "_".join(["sents",str(length),str(length_index)])
    pickle.dump(new_batch,open(os.path.join(args.outputdir,new_file),"wb"))


