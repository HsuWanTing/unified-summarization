# encoding: utf-8
import os, sys
#sys.path.append("/home/cindy/workspace/skip-thoughts")
#import skipthoughts
import cPickle
import json
import numpy as np
import nltk
import pyrouge
from pythonrouge.pythonrouge import Pythonrouge
from rouge import Rouge 
import rouge_not_a_wrapper as my_rouge
import logging
from tqdm import tqdm
import pdb


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def write_for_rouge(reference_sents, decoded_sents, ex_index, decode_dir, reference_dir):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    # make directory
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(decode_dir):
        os.makedirs(decode_dir)

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    if len(reference_sents) > 0:
        reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    dec_file = os.path.join(decode_dir, "%06d_decoded.txt" % ex_index)

    if len(reference_sents) > 0 and os.path.exists(ref_file) == False:
        #pdb.set_trace()
        with open(ref_file, "w") as f:
            for idx,sent in enumerate(reference_sents):
                f.write(sent.encode('utf-8')) if idx==len(reference_sents)-1 \
                                              else f.write(sent.encode('utf-8')+"\n")
    with open(dec_file, "w") as f:
        for idx,sent in enumerate(decoded_sents):
            f.write(sent.encode('utf-8')) if idx==len(decoded_sents)-1 \
                                          else f.write(sent.encode('utf-8')+"\n")


def rouge_eval(decode_dir, reference_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    #r.system_filename_pattern = '(\d+)_reference.txt'
    r.model_dir = reference_dir
    r.system_dir = decode_dir
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)

def rouge_log(results_dict, dir_to_write, model_name):
    """Log ROUGE results to screen and write to file.

    Args:
        results_dict: the dictionary returned by pyrouge
        dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1","2","l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x,y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str) # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results_"+model_name+".txt")
    print("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)


def get_each_summary_score(extract_dir, ref_dir, result_path):
    print '[Info] Get each summary score between extract_dir = ', extract_dir, \
          'and ref_dir = ', ref_dir
    ROUGE_path = '/home/cindy/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl'
    data_path = '/home/cindy/pyrouge/tools/ROUGE-1.5.5/data'
    rouge = Pythonrouge(n_gram=2, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, \
                        word_level=True, length_limit=False, use_cf=False, cf=95, \
                        scoring_formula="average", resampling=False, samples=1, favor=True, p=0.5)
    
    all_scores = {}
    for f_ext in tqdm(os.listdir(extract_dir)):
        ext_summ = open(os.path.join(extract_dir, f_ext)).read().splitlines()
        idx = f_ext.split('_')[0]
        ref_summ = open(os.path.join(ref_dir, idx + '_reference.txt')).read().splitlines()
        setting_file = rouge.setting(files=False, summary=[ext_summ], reference=[[ref_summ]])
        scores = rouge.eval_rouge(setting_file, f_measure_only=True, \
                                  ROUGE_path=ROUGE_path, data_path=data_path)
        all_scores[idx] = scores

    pdb.set_trace()
    cPickle.dump(all_scores, open(result_path, 'wb'))


def lead_3_summ(data, extract_dir, ref_dir):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i, story in tqdm(enumerate(data)):
        if os.path.exists(os.path.join(extract_dir, "%06d_decoded.txt" % i)):
            continue

        article = story[1].decode('utf-8')
        abstract = story[0].replace('<s>', '')
        art_sents = tokenizer.tokenize(article)
        abs_sents = [s.strip() for s in abstract.split('</s>') if len(s) > 0]
        summ = art_sents[:3]

        # write file
        write_for_rouge(abs_sents, summ, i, extract_dir, ref_dir)


def edit_dist_summ(data, extract_dir, ref_dir):

    print('run extractive summarization using edit distance')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i, story in tqdm(enumerate(data)):
        if os.path.exists(os.path.join(extract_dir, "%06d_decoded.txt" % i)):
            continue

        #article = unicode(story[1], 'utf-8')
        article = story[1].decode('utf-8')
        abstract = story[0].replace('<s>', '')
        art_sents = tokenizer.tokenize(article)
        abs_sents = [s.strip() for s in abstract.split('</s>') if len(s) > 0]
        
        summ = []
        for abs_sent in abs_sents:
            min_dist = 1000.0
            for art_sent in art_sents:
                dist = nltk.edit_distance(abs_sent.split(), art_sent.split())
                if dist < min_dist:
                    summ_sent = art_sent
                    min_dist = dist
            if summ_sent not in summ:
                summ.append(summ_sent)

        # write file
        write_for_rouge(abs_sents, summ, i, extract_dir, ref_dir)


def rouge_summ(data, extract_dir, ref_dir, ROUGE='ROUGE-L'):
    # https://github.com/tagucci/pythonrouge
    print('run extractive summarization using ROUGE scores')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    ROUGE_path = '/home/cindy/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl'
    data_path = '/home/cindy/pyrouge/tools/ROUGE-1.5.5/data'
    rouge = Pythonrouge(n_gram=1, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, \
                        word_level=True, length_limit=False, use_cf=False, cf=95, \
                        scoring_formula="average", resampling=False, samples=1, favor=True, p=0.5)

    for i, story in tqdm(enumerate(data)):
        if os.path.exists(os.path.join(extract_dir, "%06d_decoded.txt" % i)):
            continue

        article = story[1].decode('utf-8')
        abstract = story[0].decode('utf-8').replace('<s>', '')

        art_sents = tokenizer.tokenize(article)
        abs_sents = [s.strip() for s in abstract.split('</s>') if len(s) > 0]

        summ = []
        for abs_sent in abs_sents:
            max_rouge_l = 0.0
            for art_sent in art_sents:
                setting_file = rouge.setting(files=False, summary=[[art_sent.encode('utf-8')]], \
                                                          reference=[[[abs_sent.encode('utf-8')]]])
                scores = rouge.eval_rouge(setting_file, f_measure_only=True, \
                                          ROUGE_path=ROUGE_path, data_path=data_path)
                if scores[ROUGE] > max_rouge_l:
                    extract_sent = art_sent
                    max_rouge_l = scores[ROUGE]
            if extract_sent not in summ:
                summ.append(extract_sent)

        # write file
        write_for_rouge(abs_sents, summ, i, extract_dir, ref_dir)

def rouge_summ_not_wrapper(data, extract_dir, ref_dir, ROUGE='rouge-l'):
    # https://github.com/pltrdy/rouge
    print('run extractive summarization using ROUGE scores (not a wrapper)')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    rouge = Rouge()

    for i, story in tqdm(enumerate(data)):
        if os.path.exists(os.path.join(extract_dir, "%06d_decoded.txt" % i)):
            continue

        article = story[1].decode('utf-8')
        abstract = story[0].decode('utf-8').replace('<s>', '')

        art_sents = tokenizer.tokenize(article)
        abs_sents = [s.strip() for s in abstract.split('</s>') if len(s) > 0]

        summ = []
        for abs_sent in abs_sents:
            max_rouge_l = 0.0
            for art_sent in art_sents:
                scores = rouge.get_scores([art_sent], [abs_sent])
                if scores[0][ROUGE]['f'] > max_rouge_l:
                    extract_sent = art_sent
                    max_rouge_l = scores[0][ROUGE]['f']
            if extract_sent not in summ:
                summ.append(extract_sent)

        # write file
        write_for_rouge(abs_sents, summ, i, extract_dir, ref_dir)


'''
This one is the same as the function above (rouge_summ_not_wrapper)
But the input is different.
This one take the result json files as input.
The file contains a dictionary that has 3 keys: 'article', 'ref', 'gen'.
'article' is a single string, while 'ref' and 'gen' are list of sentences.
'''
def rouge_summ_not_wrapper_2(result_dir, extract_dir, ref_dir, new_result_dir, ROUGE='rouge-l'):
    # https://github.com/pltrdy/rouge
    print('run extractive summarization using ROUGE scores (not a wrapper)')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    rouge = Rouge()

    for f in tqdm(os.listdir(result_dir)):
        index = int(f.split('.')[0].split('_')[1])
        data = json.load(open(os.path.join(result_dir, f)))
        article = data['article']
        abs_sents = data['ref']
        art_sents = tokenizer.tokenize(article)

        summ = []
        for abs_sent in abs_sents:
            max_rouge_l = 0.0
            for art_sent in art_sents:
                scores = rouge.get_scores([art_sent], [abs_sent])
                if scores[0][ROUGE]['f'] > max_rouge_l:
                    extract_sent = art_sent
                    max_rouge_l = scores[0][ROUGE]['f']
            if extract_sent not in summ:
                summ.append(extract_sent)

        # write file
        data['ext'] = summ
        json.dump(data, open(os.path.join(new_result_dir, f), 'w'))
        write_for_rouge(abs_sents, summ, index, extract_dir, ref_dir)

def my_rouge_l_summ_not_wrapper_2(result_dir, extract_dir, ref_dir, new_result_dir):
    # https://github.com/pltrdy/rouge
    print('run extractive summarization using my ROUGE scores (not a wrapper)')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    rouge = Rouge()

    for f in tqdm(os.listdir(result_dir)):
        if os.path.exists(os.path.join(new_result_dir, f)):
            continue
        index = int(f.split('.')[0].split('_')[1])
        data = json.load(open(os.path.join(result_dir, f)))
        article = data['article']
        abs_sents = data['ref']
        art_sents = tokenizer.tokenize(article)

        summ = []
        for abs_sent in abs_sents:
            max_rouge_l = 0.0
            for art_sent in art_sents:
                score_f, _, _ = my_rouge.rouge_l_summary_level([art_sent], [abs_sent])
                if score_f > max_rouge_l:
                    extract_sent = art_sent
                    max_rouge_l = score_f
            if extract_sent not in summ:
                summ.append(extract_sent)

        # write file
        data['ext'] = summ
        json.dump(data, open(os.path.join(new_result_dir, f), 'w'))
        write_for_rouge(abs_sents, summ, index, extract_dir, ref_dir)

def st_extract_summ(data, extract_dir, ref_dir):

    print('run extractive summarization using skip-thoughts')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    for i, story in tqdm(enumerate(data)):
        if os.path.exists(os.path.join(extract_dir, "%06d_decoded.txt" % i)):
            continue

        article = story[1].decode('utf-8')
        abstract = story[0].replace('<s>', '')

        #art_sents = [s.strip() for s in article.split('.') if len(s) > 4]
        art_sents = tokenizer.tokenize(article)
        art_sents = [s.replace(' .', '').strip() for s in art_sents if len(s) > 5]
        abs_sents = [s.replace(' .', '').strip() for s in abstract.split('</s>') if len(s) > 0]

        #try:
        art_vecs = encoder.encode(art_sents)
        abs_vecs = encoder.encode(abs_sents)
        #except:
        #    continue
    
        extract_summ = []

        for j in range(len(abs_sents)):
            abs_vec = abs_vecs[j]
            # cosine similarities for every article sentences with abstract ith sentence
            cos_sim = np.dot(art_vecs, abs_vec) / np.sum(art_vecs*art_vecs, 1)
            extract_summ.append(art_sents[np.argmax(cos_sim)])

        # add period
        abs_sents = [s + ' .' for s in abs_sents if s[-1].isalpha()]
        extract_summ = [s + ' .' for s in extract_summ if s[-1].isalpha()]

        # write file
        write_for_rouge(abs_sents, extract_summ, i, extract_dir, ref_dir)


if __name__ == '__main__':

    data_path = 'data/cnn_dailymail/test.pkl'
    exp_dir = 'CNNDM_extract'
    ref_dir = os.path.join(exp_dir, 'reference')
    st_dir = os.path.join(exp_dir, 'st_v2')
    lead_3_dir = os.path.join(exp_dir, 'lead_3_v2')
    edit_dist_dir = os.path.join(exp_dir, 'edit_dist_v2')
    #rouge_l_dir = os.path.join(exp_dir, 'rouge_l_stem_stop')
    rouge_l_dir = os.path.join(exp_dir, 'rouge_l')
    rouge_l_not_wrapper_dir = os.path.join(exp_dir, 'rouge_l_not_wrapper_no_repeat')

    '''
    data = cPickle.load(open(data_path, 'rb'))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    '''
    #st_extract_summ(data, st_dir, ref_dir)
    #results_dict = rouge_eval(st_dir, ref_dir)
    #rouge_log(results_dict, exp_dir, 'st_v2')

    #lead_3_summ(data, lead_3_dir, ref_dir)
    #results_dict = rouge_eval(lead_3_dir, ref_dir)
    #rouge_log(results_dict, exp_dir, 'lead_3_v2')

    #edit_dist_summ(data, edit_dist_dir, ref_dir)
    #results_dict = rouge_eval(edit_dist_dir, ref_dir)
    #rouge_log(results_dict, exp_dir, 'edit_dist_v2')

    #rouge_summ(data, rouge_l_dir, ref_dir)
    #results_dict = rouge_eval(rouge_l_dir, ref_dir)
    #rouge_log(results_dict, exp_dir, 'rouge_l_stem_stop')
    #rouge_log(results_dict, exp_dir, 'rouge_l')

    #rouge_summ_not_wrapper(data, rouge_l_not_wrapper_dir, ref_dir)
    #results_dict = rouge_eval(rouge_l_not_wrapper_dir, ref_dir)
    #rouge_log(results_dict, exp_dir, 'rouge_l_not_wrapper_no_repeat')

    #get_each_summary_score(rouge_l_not_wrapper_dir, ref_dir, 'ROUGE_results_rouge_l_not_wrapper_no_repeat.pkl')

    result_dir = '/home/cindy/workspace/summarization/pointer-generator/log/cnn_dailymail/decode_train_400maxenc_4beam_35mindec_120maxdec_ckpt-236000/result'
    extract_dir = 'CNNDM_train/extract/my_rouge_l_not_wrapper_no_repeat'
    ref_dir = 'CNNDM_train/extract/reference'
    new_result_dir = 'CNNDM_train/result'
    my_rouge_l_summ_not_wrapper_2(result_dir, extract_dir, ref_dir, new_result_dir)
    results_dict = rouge_eval(extract_dir, ref_dir)
    rouge_log(results_dict, 'CNNDM_train/extract', 'my_rouge_l_not_wrapper_no_repeat')



