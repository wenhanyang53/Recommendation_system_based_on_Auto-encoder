
import nltk
import sklearn
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

nltk.download('brown')
data = pd.read_excel("merged_tra.xlsx")

id_and_desc = []
temp = []
id = data['Identifiant']
desc_list = data['Descriptif long_eng']
for i in range(0,len(data)):
    temp.append(id[i])
    temp.append(desc_list[i])
    id_and_desc.append(temp)
    temp = []
# print(id_and_desc[0])

# 1. Remove short keywords
# 2. Create a map to track how many times a keyword is repeated
# 3. Drop off all the keywords that don't repeat and set a threshold for <25% of repetition (to avoid generic words such as manifestiation, Toulouse, etc. )
# 4. Build this data to be put in the NN.
# TextBlob getting nouns from the description

def extracting_keywords(desc_list):
    total_words = 0
    array_of_keywords=[] # First item will be the identificant, then the list of keywords
    for i in id_and_desc:
        temp=[]
        temp.append(i[0]) #First item = Idenficiant
        blob = TextBlob(i[1])        
        #r.extract_keywords_from_text(i[1])
        phrases = blob.noun_phrases #r.get_ranked_phrases()
        total_words += len(phrases)
        # Got all phrases from the description
        for j in phrases:
            #print()
            if len(j) < 3:
                phrases.remove(j)               # Filter to drop a phrase of len smaller than 5
        temp.append(phrases)
        array_of_keywords.append(temp)
    # print(total_words)
    return array_of_keywords
extract_of_keyphrases = extracting_keywords(desc_list)      # This array has id + phrases 
# print(type(extract_of_keyphrases))

#Time to find the words that repeat actually in the description of diff events
record = {}
def finding_repeats(extract_of_keyphrases):
    words_list=[]
    for p in range(len(extract_of_keyphrases)):
        words_list.append(list(extract_of_keyphrases[p][1]))
    #print((words_list[0]))
    for i in words_list:
        for j in i:
            if j in record:
                record[j]+=1
            else:
                record[j] = 1
    #print(record)
    return(record)
xx = finding_repeats(extract_of_keyphrases)

# Changing type of wordlist to normal words
new_dict={}
for x in extract_of_keyphrases: 
    new_dict[x[0]] = str(x[1]).replace("'","").replace("[","").replace("]",'').split(",")
# print(new_dict)

# enc = FeatureHasher(input_type='string')
enc = FeatureHasher(input_type='string', n_features=511)

f = enc.transform(new_dict)

x = f.toarray()

len(x)

df = pd.DataFrame(x,index=new_dict)
# print(df)

# So One hot encoding on 247 rows of data (based on their ID number makes it 1048576 columns - features)
# @Wenhan
# Clustering on this is gonna be a pain

