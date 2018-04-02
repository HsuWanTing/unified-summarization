import sys, os
sys.path.append("/data/VSLab/cindy/Workspace/summarization/pointer-generator-pg-func3")
import rouge_not_a_wrapper as rouge_quick
from tqdm import tqdm
import nltk
import cPickle
import pdb

article_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/articles'
abstract_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/reference'
gen_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/pointer-gen-cov'

def rouge_scores_quick(summ, ref, N=10):
    '''
    This is the quick version of rouge, but the value may be different.
    summ: summary that need to be calculated, a list of sentences                                     
    ref: reference summary, a list of sentences
    '''
    result = {}
    for i in range(N):
        f, p, r = rouge_quick.rouge_n(summ, ref, i+1)
        result['rouge-' + str(i+1)] = {'f': f, 'p': p, 'r': r}
    return result


def get_each_summary_copy_rate(summ_dir, art_dir=None, articles=None):
    print '[Info] Get each Rouge copy rate between article and summary dir = ', summ_dir
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    all_scores = {}
    for f_summ in tqdm(os.listdir(summ_dir)):
        if f_summ[0] == '.':
            continue
        summ = open(os.path.join(summ_dir, f_summ)).read().splitlines()
        idx = f_summ.split('_')[0]
        if articles:
            article = articles[int(idx)]
        else:
            article = open(os.path.join(art_dir, idx + '_article.txt')).read()
        article = tokenizer.tokenize(article.decode('utf-8'))                          
        article = [a.encode('utf-8') for a in article]
        all_scores[idx] = rouge_scores_quick(summ, article, 10)
    return all_scores


def get_means(copy_rates):
    means = []
    N = len(copy_rates.values()[0].keys())
    for n in range(N):
        rouge_n_list = [c['rouge-' + str(n+1)]['p'] for c in copy_rates.values()]
        mean = sum(rouge_n_list) / len(rouge_n_list)
        means.append(mean)
    return means
    


if __name__ == '__main__':

    if os.path.exists('ref_copy_rates.pkl') and os.path.exists('gen_copy_rates.pkl'):
        ref_copy_rates = cPickle.load(open('ref_copy_rates.pkl'))
        gen_copy_rates = cPickle.load(open('gen_copy_rates.pkl'))
    else:
        ref_copy_rates = get_each_summary_copy_rate(article_dir, abstract_dir)
        gen_copy_rates = get_each_summary_copy_rate(article_dir, gen_dir)
        pdb.set_trace()
        cPickle.dump(ref_copy_rates, open('ref_copy_rates.pkl', 'wb'))
        cPickle.dump(gen_copy_rates, open('gen_copy_rates.pkl', 'wb'))

    ref_means = get_means(ref_copy_rates)
    gen_means = get_means(gen_copy_rates)

    print 'means of reference copy rate (rouge-1 ~ rouge-10):'
    print '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'%(ref_means[0], ref_means[1], ref_means[2], ref_means[3], ref_means[4], ref_means[5], ref_means[6], ref_means[7], ref_means[8], ref_means[9])

    print 'means of generated copy rate (rouge-1 ~ rouge-10):'
    print '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'%(gen_means[0], gen_means[1], gen_means[2], gen_means[3], gen_means[4], gen_means[5], gen_means[6], gen_means[7], gen_means[8], gen_means[9])



