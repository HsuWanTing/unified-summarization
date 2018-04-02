import sys, os
import reward_criterion
from tqdm import tqdm
#import nltk
import cPickle
import pdb

article_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/articles'
abstract_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/reference'
gen_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-gen-official-test-output/pointer-gen-cov'


def get_all_copy_rates(art_dir, summ_dir):
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    all_scores = {}
    for f_summ in tqdm(os.listdir(summ_dir)):
        if f_summ[0] == '.':
            continue
        summ = open(os.path.join(summ_dir, f_summ)).read().splitlines()
        idx = f_summ.split('_')[0]
        article = open(os.path.join(art_dir, idx + '_article.txt')).read()
        #article = tokenizer.tokenize(article.decode('utf-8'))                          
        #article = [a.encode('utf-8') for a in article]
        all_scores[idx] = reward_criterion.copy_rate_rouge_n(summ, article,1)
    return all_scores


def get_means(copy_rates):
    copy_rates_values = copy_rates.values()
    return sum(copy_rates_values) / len(copy_rates_values)
    


if __name__ == '__main__':

    ref_copy_rates = get_all_copy_rates(article_dir, abstract_dir)
    #gen_copy_rates = get_all_copy_rates(article_dir, gen_dir)

    ref_means = get_means(ref_copy_rates)
    #gen_means = get_means(gen_copy_rates)

    print 'means of reference copy rate:', ref_means
    #print 'means of generated copy rate:', gen_means



