import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import plotly.plotly as py
import json
import cPickle
import os, sys
import itertools
import pdb
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
import my_copy_rate_metric as copy_rate
import rouge_copy_rate


#decode_dir = '../log/CNNDM_300k_no_ss/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-236000/'
decode_dir = sys.argv[1]
print 'Analyze decoded result in dir = ', decode_dir
ref_dir = os.path.join(decode_dir, 'reference')
vis_dir = os.path.join(decode_dir, 'visualize')
save_path = os.path.join(decode_dir, 'analysis_result.json')

ppoint_avg = []
point_ratio = []
point_ratio_sum = 0.0
gen_words = {'article_oov': defaultdict(int), 
             'article_non_oov': defaultdict(int), 
             'novel': defaultdict(int)}
total_dec_sents = 0
total_ref_sents = 0
total_dec_short_sents = 0
dec_copy_sents = 0
ref_copy_sents = 0
error_ref_sents = 0


def analysis_copy_rate():
    ########################################
    # calculate my_copy_rate_metric
    #######################################
    
    d_path = '/data/VSLab/cindy/Workspace/summarization/analysis/reward_analysis/data/cnn_dailymail/test.pkl'
    data = cPickle.load(open(d_path, 'rb'))
    articles = [d[1] for d in data]
    '''
    articles = []
    art_f = sorted(os.listdir(os.path.join(decode_dir, 'articles')))
    for f in art_f:
        if f[0] == '.':
            continue
        articles.append(open(os.path.join(decode_dir, 'articles', f)).read())
    pdb.set_trace()
    '''

    # LCCS copy rate
    gen_copy_s = copy_rate.get_each_summary_copy_rate(articles, os.path.join(decode_dir, 'decoded'))
    #gen_copy_s = copy_rate.get_each_summary_copy_rate(articles, os.path.join(decode_dir, 'pointer-gen-cov'))
    cPickle.dump(gen_copy_s, open(os.path.join(decode_dir, 'my_copy_rate_gen.pkl'), 'wb'))
    id_sort = sorted([k for k in gen_copy_s])
    gen_copy_rates_list = [gen_copy_s[i] for i in id_sort]
    copy_rate.plot_histogram(gen_copy_rates_list, os.path.join(decode_dir, 'my_decoded_copy_rate.png'))
    print 'LCCS copy rate:', sum(gen_copy_rates_list)/len(gen_copy_rates_list)

    # Rouge1~9 copy rate
    gen_copy_s = rouge_copy_rate.get_each_summary_copy_rate(os.path.join(decode_dir, 'decoded'), articles=articles)
    #gen_copy_s = copy_rate.get_each_summary_copy_rate(articles, os.path.join(decode_dir, 'pointer-gen-cov'))
    cPickle.dump(gen_copy_s, open(os.path.join(decode_dir, 'rouge_copy_rate_gen.pkl'), 'wb'))
    gen_means = rouge_copy_rate.get_means(gen_copy_s)
    print 'means of generated copy rate (rouge-1 ~ rouge-10):'
    print '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'%(gen_means[0], gen_means[1], gen_means[2], gen_means[3], gen_means[4], gen_means[5], gen_means[6], gen_means[7], gen_means[8], gen_means[9])


def tokens2sents(tokens):
    if tokens[-1] == '[EOS]':
        text = ' '.join(tokens[:-1])
    else:
        text = ' '.join(tokens)
    sents = text.split('.')
    long_sents = []
    short_sents = []
    for s in sents:
        if len(s) > 3:
            long_sents.append((s + '.').strip())
        else:
            short_sents.append((s + '.').strip())
    #if len(short_sents) > 0:
    #    pdb.set_trace()
    return long_sents, short_sents


def analysis():

    for f in tqdm(os.listdir(vis_dir)):
        # read article and decoded data
        js = json.load(open(os.path.join(vis_dir, f)))
        article = [w.replace('__', '') for w in js['article_lst']]
        article_oovs = [w.replace('__', '') for w in js['article_lst'] if w[:2] == '__' and w[-2:] == '__']
        abstract = js['decoded_lst']
        p_gens = [p[0] for p in js['p_gens']]
        probs = js['probs']
        attns = js['attn_dists']

        # read reference
        ref_file = open(os.path.join(ref_dir, f.split('.')[0].split('_')[-1] + '_reference.txt'))
        #ref = ' '.join(ref_file.read().splitlines())
        #ref = ref.split()
        ref = ref_file.read().splitlines()
        ref = [r.decode('utf-8') for r in ref]
        ref_file.close()

        #pdb.set_trace()
        # check for point probability
        ppoint_avg.append(1 - sum(p_gens)/len(p_gens))

        # check for  point ratio
        analysis_point_ratio(article, article_oovs, abstract, probs, attns, p_gens)

        # check for copied n-gram from article
        analysis_copy_sentences(article, abstract, ref, f)

    ppoint_avg_avg = sum(ppoint_avg) / len(ppoint_avg)
    point_ratio_avg = point_ratio_sum/sum([len(p) for p in point_ratio])
    gen_words_order = {}
    gen_words_order['article_oov'] = sorted(gen_words['article_oov'].items(), key=lambda x:x[1], reverse=True)
    gen_words_order['article_non_oov'] = sorted(gen_words['article_non_oov'].items(), key=lambda x:x[1], reverse=True)
    gen_words_order['novel'] = sorted(gen_words['novel'].items(), key=lambda x:x[1], reverse=True)
    print 'point_ratio_avg: ', point_ratio_avg
    print 'ppoint_avg_avg: ', ppoint_avg_avg
    print 'reference copy sentences: ', ref_copy_sents, '/', float(total_ref_sents), '=', ref_copy_sents / float(total_ref_sents)
    print 'decoded copy sentences: ', dec_copy_sents, '/', float(total_dec_sents), '=', dec_copy_sents / float(total_dec_sents)
    if error_ref_sents > 0:
        print 'error refernce sentences: ', error_ref_sents, '/', total_ref_sents
    print 'total decoded short sentences (len <= 3): ', total_dec_short_sents


    data = {'ppoint_avg_list': ppoint_avg,
             'ppoint_avg_avg': ppoint_avg_avg,
             'point_ratio_list': point_ratio,
             'point_ratio_avg': point_ratio_avg,
             'high_gen_words': gen_words_order,
             'ref_copy_sents%': ref_copy_sents / float(total_ref_sents),
             'dec_copy_sents%': dec_copy_sents / float(total_dec_sents),
             'dec_short_sents': total_dec_short_sents}
    save_data(data)
    return data



def save_data(data):
    json.dump(data, open(save_path, 'w'))

def read_file(path):
    data = json.load(open(path))
    return data

def analysis_point_ratio(article, article_oovs, abstract, probs, attns, p_gens):
    global point_ratio
    global point_ratio_sum
    global gen_words
    point_ratio_ = []
    if len(abstract) < len(probs):
        abstract.append('[EOS]')
    for i,p in enumerate(probs):
        point_p = sum([att for j,att in enumerate(attns[i]) if article[j] == abstract[i]]) * (1-p_gens[i])
        if point_p/p > 1.01:
            point_ratio_.append(1.0)
        else:
            point_ratio_.append(point_p/p)

        if point_p/p < 0.05:
            if abstract[i] in article_oovs:
                gen_words['article_oov'][abstract[i]] += 1
            elif abstract[i] in article:
                gen_words['article_non_oov'][abstract[i]] += 1
            else:
                gen_words['novel'][abstract[i]] += 1
    
    point_ratio.append(point_ratio_)
    point_ratio_sum += sum(point_ratio_)

def analysis_copy_sentences(article, abstract, ref, fname):
    global total_dec_sents
    global total_ref_sents
    global total_dec_short_sents
    global dec_copy_sents
    global ref_copy_sents
    global error_ref_sents

    abs_sents, abs_short_sents = tokens2sents(abstract)
    article_txt = ' '.join(article)
    total_dec_sents += len(abs_sents)
    total_ref_sents += len(ref)
    total_dec_short_sents += len(abs_short_sents)
    #pdb.set_trace()
    for s in abs_sents:
        try:
            if s in article_txt:
                dec_copy_sents += 1
        except:
            print fname, ':', s
    for s in ref:
        try:
            if s in article_txt:
                ref_copy_sents += 1
        except:
            error_ref_sents += 1
            print fname.split('.')[0].split('_')[-1] + '_reference.txt', ':', s


def plot_ppoint(ppoint_avg):
    plt.figure(figsize=(9, 5))
    plt.hist(np.array(ppoint_avg), bins=np.arange(0.0, 1.0 + 0.05, 0.05))
    plt.title("Average Point Probability")
    plt.xlabel("average point probability")
    plt.ylabel("number of abstracts (total %d)" % len(ppoint_avg))
    plt.xlim(0.2, 1.0)
    plt.xticks(np.arange(0.2, 1.0+0.05, 0.05))
    plt.grid(True)
    plt.savefig(os.path.join(decode_dir, 'ppoint.png'))
    plt.clf()

def plot_point_ratio(point_ratio):
    point_ratio_all = list(itertools.chain.from_iterable(point_ratio))
    plt.hist(np.array(point_ratio_all), bins=np.arange(0.0, 1.0 + 0.05, 0.05))
    plt.title("Point Ratio of Predicted Words")
    plt.xlabel("point ratio (attention prob. / word prob.)")
    plt.ylabel("number of words (total %d)" % len(point_ratio_all))
    plt.xlim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.0+0.05, 0.05))
    plt.grid(True)
    plt.savefig(os.path.join(decode_dir, 'point_ratio.png'))
    plt.clf()

def plot_abstract_len(point_ratio):
    abs_len = [len(p) for p in point_ratio]
    plt.hist(np.array(abs_len), bins=np.arange(30, 125 + 5, 5))
    plt.title("Abstract Length")
    plt.xlabel("length")
    plt.ylabel("number of abstract (total %d)" % len(abs_len))
    #plt.xlim(30,130)
    #plt.ylim(0,4000)
    plt.xticks(np.arange(30, 125+5, 5))
    plt.grid(True)
    plt.savefig(os.path.join(decode_dir, 'abs_len.png'))
    plt.clf()



if __name__=='__main__':
    
    if os.path.isfile(save_path):
        data = read_file(save_path)
    else:
        data = analysis()

    plot_ppoint(data['ppoint_avg_list'])
    plot_point_ratio(data['point_ratio_list'])
    plot_abstract_len(data['point_ratio_list'])
    
    analysis_copy_rate() 


