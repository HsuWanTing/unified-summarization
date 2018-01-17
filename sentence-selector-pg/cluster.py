import string
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from collections import Counter
#from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import cPickle as pk
import pdb
#import data
import math
from tqdm import tqdm
 
 
def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    #text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode('utf-8').translate(None, string.punctuation).decode('utf-8')
    tokens = text.split()
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=1.0,
                                 min_df=1,
                                 lowercase=True)

    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters, n_init=100, verbose=0, tol=1e-10)
    km_model.fit(tfidf_model)
    #print 'inertia: ', km_model.inertia_
    #pdb.set_trace()
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering
 
 
if __name__ == "__main__":

    #test_data = pk.load(open('/data/VSLab/cindy/Workspace/summarization/data/CNN_Dailymail/Rouge_scores/test_scores.pkl'))
    data = pk.load(open('/data/VSLab/cindy/Workspace/summarization/sentence-selector/log/exp11_exp9_new_avg_pool/select_test_100maxart_50maxsent_ckpt-54130/probs.pkl'))
    cluster_num = int(sys.argv[1])
    print 'cluster num: ', cluster_num

    output_file = os.path.join('log', 'gt_diversity_%dcluster.pkl' % cluster_num)
    if os.path.exists(output_file):
        print 'result file exists, load ', output_file
        save_data = pk.load(open(output_file))
        diversity = save_data['diversity']
    else:
        # min diversity: 
        # 3 clusters: 0.577 (0.33)
        # 4 clusters: 0.5   (0.25)
        # 5 clusters: 0.447 (0.2)
        diversity = []
        clusters_all = []
        for key in tqdm(data['article']):
            #sentences = data.abstract2sents(article)
            if len(data['article'][key]) < cluster_num:
                diversity.append(0)
                clusters_all.append({c:idx for c, idx in enumerate(data['article'][key])})
                continue
            clusters = cluster_texts(data['article'][key], cluster_num)
            #pprint(dict(clusters))
            gt_ids = data['gt_ids'][key]
            counter = {c:0 for c in clusters}
            for idx in gt_ids:
                for c in clusters:
                    if idx in clusters[c]:
                        counter[c] += 1
            ratio = [float(counter[c])/len(gt_ids) for c in counter]
            div = sum([r**2 for r in ratio])
            diversity.append(div)
            clusters_all.append(clusters)

        save_data = {'diversity': diversity, 'clusters': clusters_all}
        pk.dump(save_data, open(output_file, 'wb'))

    if cluster_num == 5:
        min_div = 0.2
    elif cluster_num == 4:
        min_div = 0.25
    elif cluster_num == 3:
        min_div = 0.3

    avg_diversity = sum(diversity) / len(diversity)
    plt.figure(figsize=(18, 8))
    plt.hist(np.array(diversity), bins=np.arange(min_div, 1.0001, 0.02))
    plt.xticks(np.arange(min_div, 1.0001, 0.02))
    plt.grid(True)
    plt.xlabel("diversity")
    plt.ylabel("number of articles")
    plt.title("Diversity of GT selected sentences (%d clusters)" % cluster_num)
    plt.savefig(os.path.join('log', 'gt_diversity_%dcluster_2.png' % cluster_num))
    plt.clf()
    pdb.set_trace()
