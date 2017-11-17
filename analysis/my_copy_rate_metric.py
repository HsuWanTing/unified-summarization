import os
import cPickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


decoded_dir = '/data/VSLab/cindy/Workspace/summarization/pointer-generator-ss/pgen_is_0.5/CNNDM_300k_ss0.75/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-236000'
extract_dir = 'CNNDM_extract/rouge_l_not_wrapper_no_repeat'
abstract_dir = 'CNNDM_pointer_gen/reference'

def _lcs(x, y):
    n, m = len(x), len(y)
    lcs_len = 0
    #cs_len = 0
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
                if table[i, j] > lcs_len:
                    lcs_len = table[i, j]
            else:
                table[i, j] = 0
    return lcs_len


def copy_rate(summary, source):
    # summary is a list of sentences
    # source is a single string
    lcs_total = 0
    summary_len = 0
    for s in summary:
        lcs_total += _lcs(s.split(), source.split())
        summary_len += len(s.split())
    return float(lcs_total) / summary_len


def get_each_summary_copy_rate(article_data, summ_dir):
    print '[Info] Get each LCCS copy rate between article and summary dir = ', summ_dir

    all_scores = {}
    for f_summ in tqdm(os.listdir(summ_dir)):
        if f_summ[0] == '.':
            continue
        summ = open(os.path.join(summ_dir, f_summ)).read().splitlines()
        idx = f_summ.split('_')[0]
        article = article_data[int(idx)]
        score = copy_rate(summ, article)
        all_scores[idx] = score
    return all_scores


def save_all_copy_rate(articles, abstract_dir, extract_dir, decoded_dir):
    # get copy scores
    ext_copy_s = get_each_summary_copy_rate(articles, extract_dir)
    abs_copy_s = get_each_summary_copy_rate(articles, abstract_dir)
    gen_copy_s = get_each_summary_copy_rate(articles, decoded_dir)
    all_copy_scores = {'gen_copy_rate': gen_copy_s,
                       'abs_copy_rate': abs_copy_s,
                       'ext_copy_rate': ext_copy_s,
                       'abstract_dir': abstract_dir,
                       'extract_dir': extract_dir,
                       'decoded_dir': decoded_dir}
    return all_copy_scores


def plot_histogram(values, save_path):
    #plt.figure(figsize=(9, 5))
    plt.hist(np.array(values), bins=np.arange(0.0, 1.1 + 0.1, 0.1))
    #plt.hist(np.array(rouge_l))
    #plt.title("ROUGE-L scores")
    plt.xlabel("copy rate")
    plt.ylabel("number of summaries (total %d)" % len(values))
    #plt.xlim(0.2, 1.0)
    plt.xticks(np.arange(0.0, 1.1+0.1, 0.1))
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()




if __name__ == '__main__':

    d_path = 'data/cnn_dailymail/test.pkl'
    data = cPickle.load(open(d_path, 'rb'))
    articles = [d[1] for d in data]

    # get all copy rate
    '''
    if os.path.exists('my_copy_rate.pkl'):
        copy_results = cPickle.load(open('my_copy_rate.pkl', 'rb'))
        ext_copy_rate = copy_results['ext_copy_rate']
        gen_copy_rate = copy_results['gen_copy_rate']
        abs_copy_rate = copy_results['abs_copy_rate']
    else:
        copy_results = save_all_copy_rate(articles, abstract_dir, extract_dir, decoded_dir)
        ext_copy_rate = copy_results['ext_copy_rate']
        gen_copy_rate = copy_results['gen_copy_rate']
        abs_copy_rate = copy_results['abs_copy_rate']
        cPickle.dump(copy_results, open('my_copy_rate.pkl', 'wb'))


    id_sort = sorted([k for k in copy_results['abs_copy_rate']])
    ref_copy_rates_list = [abs_copy_rate[i] for i in id_sort]
    ext_copy_rates_list = [ext_copy_rate[i] for i in id_sort]
    gen_copy_rates_list = [gen_copy_rate[i] for i in id_sort]

    plot_histogram(ref_copy_rates_list, 'my_reference_copy_rate.png')
    plot_histogram(gen_copy_rates_list, 'my_decoded_copy_rate.png')
    plot_histogram(ext_copy_rates_list, 'my_extract_copy_rate.png')
    '''
    gen_copy_s = get_each_summary_copy_rate(articles, os.path.join('decoded', decoded_dir))
    cPickle.dump(gen_copy_s, open(os.path.join(decoded_dir, 'my_copy_rate_gen.pkl'), 'wb'))
    id_sort = sorted([k for k in gen_copy_s])
    gen_copy_rates_list = [gen_copy_s[i] for i in id_sort]
    plot_histogram(gen_copy_rates_list, os.path.join(decoded_dir, 'my_decoded_copy_rate.png'))




