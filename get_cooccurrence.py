import nltk
import numpy as np

# filename = "/Users/sahaj/Desktop/py/Text_Analysis/book_text-files/book1.txt"
fprefix = "hpb1"
charactergroups = {"Harry": "Humans.Harry Potter",
						"Hermione": "Humans.Hermione Granger",
						"Ron": "Humans.Ron Weasley",
						"Dumbledore": "Humans.Albus Dumbledore",
						"Snape": "Humans.Severus Snape",
						"Malfoy": "Humans.Draco Malfoy",
						"Voldemort": "Humans.Lord Voldemort"}
parseword = "Chapter"

outputfile = fprefix+'.json'
characters = list(charactergroups.keys())

# cutoff = 15

text = open(filename).read()

orig = text.split(parseword)
numchaps = len(orig)
lenchaps = []

occurrences = dict([[char, []] for char in characters])
all_sentences = []


counter = 0
for si, s in enumerate(orig):
	sentences = nltk.sent_tokenize(s)
	for senti, sentence in enumerate(sentences):
		counter += 1
		sentence = " ".join(sentence.splitlines())
		all_sentences.append(sentence)
		tokens = nltk.word_tokenize(sentence)
		for char in characters:
			if char in tokens or char.upper() in tokens:
				occurrences[char].append(counter)
	lenchaps.append(counter)

f = open(fprefix+"_sentences.txt", "w")
f1 = open(fprefix+"_characters.txt", "w")
f.write("index\tsentence\n")
f1.write("index\t")
for char in characters:
	f1.write(char+"\t")
f1.write("\n")
for i, s in enumerate(all_sentences):
	f.write(str(i)+"\t"+s+"\n")
	f1.write(str(i)+"\t")
	for char in characters:
		if i in occurrences[char]:
			f1.write("1\t")
		else:
			f1.write("0\t")
	f1.write("\n")
f.close()
f1.close()

cooccurrence = np.zeros((len(orig), len(characters), len(characters)))
nw = 5
for char1 in characters:
	for char2 in [char for char in characters if char != char1]:
		chapcounter = 0
		for si, s in enumerate(all_sentences):
			if si>lenchaps[chapcounter]:
				chapcounter+=1
			if chapcounter>0:
				left_si = np.max([si-nw, lenchaps[chapcounter-1]])
			else:
				left_si = np.max([si-nw, 0])
			right_si = np.min([si+nw, lenchaps[chapcounter]])
			if (set(range(left_si, right_si)) &  set(occurrences[char1])) and (set(range(left_si, right_si)) &  set(occurrences[char2])):
				cooccurrence[chapcounter, characters.index(char1), characters.index(char2)] += 1

cooccurrence = np.ceil(cooccurrence/float(nw*2))

# cumcurr = cooccurrence.cumsum(0)
# cooccurrence[cooccurrence>cutoff] = cutoff
cumcurr = cooccurrence

totalcurr = cooccurrence.sum(0)

output = []
charnames = ["root."+charactergroups[char] for char in characters]


for sent in cumcurr:
	new_list = []
	for ci, char1 in enumerate(characters):
		links = []
		thickness = []
		for ci2, char2 in enumerate(characters):
			if char2 != char1:
				if totalcurr[characters.index(char1), characters.index(char2)]>0:
					links.append("root."+charactergroups[char2])
					thickness.append(int(sent[characters.index(char1), characters.index(char2)]))
		new_list.append({"name": charnames[ci],
						"connections": links,
						"thickness": dict(list(zip(links, thickness)))})
	output.append(new_list)



import json
with open(outputfile, 'w') as outfile:
  json.dump(output, outfile)
