
# coding: utf-8

# I analyzed all the text from the Harry Potter series using tf-idf, sentiment analysis, and LDA to determine how word importance and topics vary during character interactions throughout the series. For the purposes of this project, I define a character "interaction" as the mention of two character names within 50 words of each other (not including "stop-words"). Specifically, I focused on the storyline between the characters of Harry and Snape.

# #### Data cleaning and preprocessing

import pandas as pd
import csv
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import itertools
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import lda
import seaborn as sns
import matplotlib.pyplot as plt,mpld3
import mpld3
from collections import defaultdict
from nltk.corpus import stopwords





stop = [str(s) for s in stopwords.words('english')]
punc = ['.', ',', '(', ')', "'", '"','...','?','!','-','--']
stop+=punc


# Each book has slightly different formatting, these functions clean the text and split the book by chapter:




def read_book(book):
    b = csv.reader(open("book_text-files/{}.txt".format(book),"rt"))
    b = [[word.strip() for word in row] for row in b]
    b_lis = [y.replace("'",'').replace('\n',' ') for x in b for y in x]
    b_lis = ' '.join(b_lis)
    b_lis = b_lis.replace('\xe2\x80\x99',"'").replace('\xe2\x80\x9c','"').replace("'",'').replace('\xe2\x80\x9d','"')    .replace('\xe2\x80\xa6','...').replace('\xe2\x80\x94','-').replace('\xe2\x80\x98',"'").replace('\xe2\x80\xa2',' ')    .replace('\x0c','').replace('\xc2\xa6','').replace('\xc3\xa9','').replace('\xe2\x80\x93','')
    if book in ('book1','book2','book3','book4','book5'):
        b_lis = b_lis.split('CHAPTER ')
    elif book in ('book6','book7'):
        b_lis = b_lis.split('Chapter ')
    if book in ('book1','book2','book3','book6'):
        b_lis = [sentence[sentence.find(' '):] for sentence in b_lis[1:]]
    elif book in ('book4','book5'):
        b_lis = [sentence[sentence.find('- ')+1:] for sentence in b_lis[1:]]
    b_lis = [i.lower() for i in b_lis if not i.isdigit()]
    return b_lis


# These data frames have the books separated by chapter:



book1 = pd.DataFrame(read_book('book1'),columns=['chapter'])
book2 = pd.DataFrame(read_book('book2'),columns=['chapter'])
book3 = pd.DataFrame(read_book('book3'),columns=['chapter'])
book4 = pd.DataFrame(read_book('book4'),columns=['chapter'])
book5 = pd.DataFrame(read_book('book5'),columns=['chapter'])
book6 = pd.DataFrame(read_book('book6'),columns=['chapter'])
book7 = pd.DataFrame(read_book('book7'),columns=['chapter'])

books = [book1,book2,book3,book4,book5,book6,book7]


# Names and combinations for interaction analysis:



names = ['harry','ron','hermione','dumbledore','snape','malfoy','voldemort']

combos = list(itertools.combinations(names,2)) #all name combinations


# Initializing data frames to hold character mention counts:



b1_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book1))])
b2_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book2))])
b3_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book3))])
b4_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book4))])
b5_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book5))])
b6_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book6))])
b7_df = pd.DataFrame(columns=combos,index=[i for i in range(len(book7))])

combo_dfs = [b1_df,b2_df,b3_df,b4_df,b5_df,b6_df,b7_df]


# This function counts the number of times 2 character names appear within 50 (non-"stop") words of each other and adds them to a dataframe by book and chapter:



def get_counts(book,chap,name1,name2):
    words = [w for w in word_tokenize(books[book].iloc[chap,0]) if w not in stop if w.isalpha()==True]
    if name1 in words and name2 in words:
        name1_indexes = [index for index, value in enumerate(words) if value == name1]
        name2_indexes = [index for index, value in enumerate(words) if value == name2]
        distances = [abs(item[0] - item[1]) for item in itertools.product(name1_indexes, name2_indexes)]
        highest=[i for i in range(len(distances)) if distances[i]<50]
        combo_dfs[book].iloc[chap,combos.index((name1,name2))]=len(highest)
    else:
        combo_dfs[book].iloc[chap,combos.index((name1,name2))]=1


# Adding proximity counts to dataframes:



for b in range(len(books)):
    for chap in range(len(books[b])):
        for combo in combos:
            get_counts(b,chap,combo[0],combo[1])


# #### Formatting nodes and edge weights for network graphs:



nodes = pd.Series(names)
nodes_dict = nodes.to_dict()

nodes_dict = dict(zip(nodes_dict.values(),nodes_dict.keys()))

nodes_df = nodes.reset_index()
nodes_df.columns = ['Id', 'name']



def get_edges(bookdf):
    edges = bookdf.sum()
    edges_weights = edges.to_frame(name='Weight').reset_index()
    edges_weights.columns = ['Label', 'Weight']
    edges_weights['Source_Label'] = edges_weights.Label.apply(lambda x: x[0])
    edges_weights['Target_Label'] = edges_weights.Label.apply(lambda x: x[1])
    edges_weights['Source'] = edges_weights.Source_Label.apply(lambda x: nodes_dict.get(x))
    edges_weights['Target'] = edges_weights.Target_Label.apply(lambda x: nodes_dict.get(x))
    return edges_weights


# Print JSON for network graph file:


def get_json(bookdf):
    print ('{ \n "nodes":[')
    for i in range(len(nodes_df)):
        print ('{' + '"name":"{}",'.format(nodes_df.iloc[i,1]) + '"group":{}'.format(i) + '},')
    print ('],')
    print ('"links":[')
    for i in range(len(get_edges(bookdf))):
        print ('{') + ('"source":{},').format(get_edges(bookdf).iloc[i,4]) + ('"target":{},').format(get_edges(bookdf).iloc[i,5]) + ('"value":{}').format(int(get_edges(bookdf).iloc[i,1]) if int(get_edges(bookdf).iloc[i,1])>0 else 1) + ('},')
    print (']\n}')


# #### Collecting TextBlob sentiment scores for each text segment mentioning "harry" and "snape":

# This function returns only the text between positions where the 2 character names appear within 50 (non-"stop") words of each other spliced together:



def get_positions(book,chap,name1,name2):
    words = [w for w in word_tokenize(books[book].iloc[chap,0]) if w not in stop if w.isalpha()==True]
    if name1 in words and name2 in words:
        name1_indexes = [index for index, value in enumerate(words) if value == name1]
        name2_indexes = [index for index, value in enumerate(words) if value == name2]
        pairs=[(item[0],item[1]) for item in itertools.product(name1_indexes, name2_indexes) if abs(item[0] - item[1])<50]
        if pairs!=[]:
            return [x for y in [words[min(pairs[i]):max(pairs[i])+1] for i in range(len(pairs))] for x in y]
        else:
            return (0,0)
    else:
        return (0,0)



sentiment = []
harrysnape = defaultdict(list)
for b in range(len(books)):
    for chap in range(len(books[b])):
        if get_positions(b,chap,'harry','snape')==(0,0):
            sentiment.append(0)
        else:
            sentiment.append(TextBlob(' '.join(get_positions(b,chap,'harry','snape'))).sentiment.polarity)
            harrysnape[b].append(' '.join(get_positions(b,chap,'harry','snape')))


# #### Processing TF-IDF word rankings (for n-grams of size 1-6) for text segments between "harry" and "snape":


splice = []
splice+=[' '.join(v) for v in harrysnape.values()]



vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,6))
doc_vectors = vectorizer.fit_transform(splice)
mat_array = doc_vectors.toarray()
fn = vectorizer.get_feature_names()




topwords  =pd.DataFrame()
for i,l in enumerate(mat_array):
    topwords[i] = [(fn[x],l[x]) for x in (l*-1).argsort() if 'harry' not in fn[x].split() if 'snape' not in fn[x].split()][:80]




topwords.head()



# #### Generating topics (using LDA) on the text where 2 characters appear together:



def get_topics(book,name1,name2):
    for chap in range(len(books[book])):
        if get_positions(book,chap,name1,name2)!=(0,0):
            words = get_positions(book,chap,name1,name2)
            vectorizer = CountVectorizer(stop_words = "english")
            doc_vecs = vectorizer.fit_transform(words)
            vocab = vectorizer.get_feature_names()

            model = lda.LDA(n_topics=4, n_iter=300,random_state=1)
            model.fit(doc_vecs)
            topic_word = model.topic_word_
            doc_topic=model.doc_topic_
            n_top_words = 10

            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Book {} Chapter {} Topic {}: {}'.format(book+1, chap+1, i+1, ' '.join(topic_words)))


# Topics selected that adequately represent interactions/theme between Harry and Snape:



b1topics = ['','','','','','','snape talking asked looking teaches \n doesnt potions got painfully struggled',
          'snape said know idea like professor \n wrong think quite dislike',
          'snape glasses speak father quidditch \n slytherin hear staring decent joking',
         'professor troll points youre white \n mind fury cold standing sat',
         'harry snapes face shut bandages \n knocked quickly gulped nerves read',
         'snape ron mirror know room library \n tonight close got better',
         'ron hermione stone didnt theyd \n little severus possibly hed torture',
         'snape yeah knows hagrid spell \n beamed spose ive temper sweeping',
         'harry ron paced waiting quirrell \n time heard didnt promised gambled',
          'hermione ron came didnt dont \n wailed idea thats eyes turned',
         'snape quirrell away make possession \n happened sent things used fathers']
b2topics = ['missed wizard classes snape secret \n witchcraft wizardry stomachache castle hogwarts','','','',
          'snape said house black cold fairer \n train deep sidekick strict','','','',
         'harry potions office hurried fast \n severus sense thumping darkened friends','',
         'harry looking floor face professor \n ear neville closer snapes turned','',
         'harry riddle door temper pulled \n hand school sitting pushed bacon',
         'snape activity moment wandered \n seen harry noise guard thankfully corridors',
         'snape ron dean rear pruning \n professor patch sprout marched going',
         'said lockhart looking teach \n dumbledore chamber notice hatred flew shall','','']
b3topics = ['detention holiday nasty spellbooks \n years street broomstick cauldron garden seized','','','',
         'snape anger startled twisting \n hissed ron time teachers best eyes','',
         'said malfoy potion didnt ingredients \n wouldnt sir looked corner great',
         'snape goblet grindylow cauldronful \n didnt showing wandering backed room black',
         'professor sit didnt severus got \n expect eyes hes late dungeons',
         'malfoy crocodile dementor hit \n harry snape spent falling caused class',
         'ron large mcgonagall looking \n years topped year gunshot noisemaker reluctantly',
         'finishing seriously hermione \n essay remind undetectable half row getting magical','',
         'snape said didnt hogsmeade \n snapes know hed inside silent say','',
         'told head sank cloak confusing \n low sit pretense potions opportunity','',
         'harry sirius doesnt got \n quidditch didnt disliked jamess looked led',
         'black potter peter wasnt door \n killed shouted looking sirius finish',
         'snape pettigrew said abruptly \n flung wand arm lolling sirius drift',
         'minister dumbledore make sirius \n madam look dementors dont havent know',
         'snape harry shrieked pointing \n behavior saved staircase dont listened failing']
b4topics = ['','','','','','','','','','','',
          'snape harrys matched possible \n loathing hogwarts table large sinistras hat',
         'snape air dungeons marched moments \n fallen chat harry growled friend',
         'moody arts snapes mealtimes overt \n really shown job wary scared',
         'said snape cedric thing writing \n harry month ill hufflepuff horrors','',
         'snape watching ask impatient \n calmly disbelief ignoring asked went right',
         'snape knew test stinks sight \n table looking brew curse glittering','',
         'harry snape unprepared worst \n slightly enemy emerging dragons arent turned','',
         'harry potter arrived bent throwing \n supposed crossing delinquent hes itll',
         'snape harry snarled went goatee \n black billowing whats slightly wand','',
         'harry stairs potter straight \n crouch slowly norris theres closed avoid',
         'said loathed ron landed flew \n school kill saving thing banished',
         'snape said snapes potion eyes \n look beetles turned saw stolen',
         'harry crouch forest professor \n right mind angrily denying smile way',
         'snape mouse hunting potter \n returning wanted whats beak trouble headmaster',
         'harry matter know angry tone \n stopped wanted strongly personal lord',
         'harry saw thoughts talking night \n stared evening hands minutes reeling','','','',
         'harry professor potter map eyes \n father wand place hair lying',
         'said spoke finally severus \n wordlessly swept watched dog slightly need',
         'harry difficult unpleasant away \n returned professor karkaroff madame hagrid truly']
b5topics = ['','','','snape door professor ron cautiously \n told head string dammit thank','',
         'snape attempt sneak conversation \n recalled flitted gleaned weasley glimpses brief','','','','','',
         'harry taunt cauldron class draught \n face laughed slytherins yes loved',
         'staircase possibility ron fell \n right walked answers dawned lurking saw','',
         'harry drowning dreaming desperately \n date ok add essay looking questioned',
         'said youd defence arts frowned meant \n kind snape ron threatening',
         'harry potter class oh seats \n copies detention burned dropped fungi','',
         'want ive hand slytherin quite \n hit time believe refused match','','','','',
         'snape know lord lessons think \n black attempt ron let cold',
         'snape trying harrys isnt \n corridor inside snappishly open dreams door',
         'said potter dark snapes looking \n repeated evenings voice liquid blinked',
         'harry professor trouble terrible \n cut started da expulsion save word',
         'snapes hard looking looked moved \n right sir heart floor mind',
         'said james yeah ok eyes pretty \n bit didnt understand going',
         'snape sank ask dramatic hear \n think talk truthfully ended reason',
         'ooclumency potions thoroughly \n tense proceedings sitting classes tables seen dreadful',
         'harry shouted wish cried hand \n tell walking bow voldemorts school','','','','',
         'harry sirius lessons hurt explained \n yelled voldemort pleasure spat easier',
         'snape felt rush rang leading \n coldly fingers knows away fell']
b6topics = ['','snape murdered sorcerers narcissa \n high pocket prophecy reasons dumbledores sat','',
         'harry think asked say slightly \n thought yes ought potions silence','','','',
         'harry hall minutes dark gateau \n bumped slughorn flying bursting torment',
         'snape repel thought nonverbal \n gaze face raised reacted speaking strong',
         'awkwardly snape began evening \n sit instead belt enjoyable door headmistresses',
         'snape detention ive im yes \n bustled hopefully got postponed moment',
         'snape said use professor told \n levicorpus woken hand effective gave',
         'snape pomfrey knows arts dark \n yes mungos katie asked hopeful','',
         'snape like way investigating \n looking seen wanted better certain low',
         'said pretending dumbledore saying \n trust offering malfoy times saw dad',
         'harry professor party sat glaring \n suggest answer finished speak silence',
         'harry lesson throats teacher \n flitwick bent colourless sparing slytherin instructor',
         'snape heard slytherins lost care \n hard bit shuffled urged seat','',
         'harry hermione room essay prince \n finally expected low marks stupid','','',
         'snape book nickname copy eyes \n stomach know liar firmly close',
         'harry prophecy face saturday \n time looked james long waves good',
         'healed said snape silver time \n leaving merely fade raised younger',
         'harry said dumbledore time \n professor experienced fell stop doesnt didnt',
         'harry snape killed dumbledore \n hagrid death yelled malfoy uncomprehending gone',
         'said pomfrey death eaters james \n fault dead came believed chair',
         'snape evil knew bed seeing \n quietly sleeping sleep telling managed']
b7topics = ['voldemort said place assure \n table controlled body went mistrust shall',
          'dumbledores scene doubt hogwarts \n legion read fell leads castle severus','','',
         'snape george snapes say stan \n broom flying hes hushed kindly',
         'snape place using insane knives \n maneuver protection set shaky talk','','',
         'making hermione like really \n took scare house worked time curtains',
         'snape information phineas way \n thought looking contained hed handed gaze',
         'said reckon yes tell hes curse \n make figure raising rose',
         'snape know friends killed room \n good accept mcgonagall newspaper dumbledores','','',
         'snape harry nigellus professor \n told pacing idiot eccentricities sudden important',
         'ideal informer kind glad ginny \n gatherings reinstated impertinent controlled hard','','','','','',
         'asked wouldnt fact lulled putting \n feel security country want false',
         'bared dare gripped release gaze \n snape forced harry thing tightly',
         'snape said moaned possess lake \n shortly path dumbledore grief black','','',
         'snape ron hermione caught hogwarts \n killings seen scrambling getting warning',
         'limping die lookout frowning picture \n hermione isnt taller charge deputies',
         'dumbledore secrets thought albus \n natural fast telling course means army',
         'mcgonagall wand thought looked \n heard happened touch believed swiftness balance',
         'professor hermione shouted relief \n dirty large ran corner stay chamber',
         'harry wand left air voice think \n silvery reassure pulled trying',
         'snape said boy little asked eyes \n youre tunnel dissolved saw',
         'snape home girl kneeling injured \n huge holding harry hand ginny',
         'didnt end meant moaned time \n gradually happen work intended bit',
         'dumbledores voldemort chose \n plan truly ago deathstick destiny saw try']
booktopics = [b1topics,b2topics,b3topics,b4topics,b5topics,b6topics,b7topics]


# #### Creating a "rolling mean" for sentiment scores to establish a timeline showing sentiment and topics each chapter:


df = pd.DataFrame(sentiment,columns=['score'])
df['rolling'] = pd.rolling_mean(df['score'],window=35,min_periods=1,center=True)
df.to_csv('test.csv')




alltopics = []
alltopics.extend(t for t in booktopics)
alltopics = [x for y in alltopics for x in y]
alltopics = [str(i) for i in alltopics]
alltopics = ['No interaction' if x=='' else x for x in alltopics]




lens = [0, 17, 35, 57, 94, 132, 162, 199]

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_axis_bgcolor('white')
y = list(df['rolling'])
ax.set_title('Interaction Sentiment and Topics over Time: Harry and Snape',fontsize=26)
ax.set_xlabel('Books by chapter',fontsize=15)
ax.set_ylabel('Sentiment Score (moving average)',fontsize=15)
colors=['r','orange','y','g','c','b','k']
booklist=["Book 1: Sorcerer's Stone","Book 2: Chamber of Secrets","Book 3: Prisoner of Azkaban","Book 4: Goblet of Fire",
          'Book 5: Order of the Phoenix',"Book 6: Half Blood Prince","Book 7: Deathly Hallows"]
for i in range(1,8):
    lines=ax.plot(range(lens[i-1],lens[i]),y[lens[i-1]:lens[i]],marker='o',c=colors[i-1],label=booklist[i-1])
    mpld3.plugins.connect(fig, mpld3.plugins.PointLabelTooltip(lines[0],labels=alltopics[lens[i-1]:lens[i]+1],voffset=20,hoffset=30))
ax.plot(range(200),[0]*200,'-',color='k')
ax.yaxis.grid()
ax.yaxis.labelpad = 10
plt.legend(loc="upper left",prop={'size':13})
plt.tight_layout()
mpld3.display()
print(mpld3.fig_to_html(fig,use_http=True))
# mpld3.save_html(fig,'hs.html')
