from pythonrouge.pythonrouge import Pythonrouge
import rouge_not_a_wrapper as rouge_quick
import pdb

def rouge_scores(summ, ref):
    '''
    summ: summary that need to be calculated, a list of sentences
    ref: reference summary, a list of sentences
    '''
    ROUGE_path = '/data/VSLab/cindy/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl'
    data_path = '/data/VSLab/cindy/pyrouge/tools/ROUGE-1.5.5/data'
    rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, ROUGE_W=False, \
                        stemming=False, stopwords=False, word_level=True, \
                        length_limit=False, use_cf=False, cf=95, scoring_formula="average", \
                        resampling=False, samples=1, favor=True, p=0.5)

    setting_file = rouge.setting(files=False, summary=[summ], reference=[[ref]])

    scores = rouge.eval_rouge(setting_file, f_measure_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
    return scores['ROUGE-1'], scores['ROUGE-2'], scores['ROUGE-L']
    '''
    scores = rouge.eval_rouge(setting_file, f_measure_only=False, ROUGE_path=ROUGE_path, data_path=data_path)
    return scores['ROUGE-1-F'], scores['ROUGE-1-P'], scores['ROUGE-1-R'],\
           scores['ROUGE-2-F'], scores['ROUGE-2-P'], scores['ROUGE-2-R'],\
           scores['ROUGE-L-F'], scores['ROUGE-L-P'], scores['ROUGE-L-R']
    '''

def rouge_scores_quick(summ, ref, metric=['rouge-1', 'rouge-2', 'rouge-l']):
    '''
    This is the quick version of rouge, but the value may be different.
    summ: summary that need to be calculated, a list of sentences
    ref: reference summary, a list of sentences
    '''
    result = {}
    if 'rouge-1' in metric:
      rouge_1_f, rouge_1_p, rouge_1_r = rouge_quick.rouge_n(summ, ref, 1)
      result['rouge-1-f'] = rouge_1_f
    if 'rouge-2' in metric:
      rouge_2_f, rouge_2_p, rouge_2_r = rouge_quick.rouge_n(summ, ref, 2)
      result['rouge-2-f'] = rouge_2_f
    if 'rouge-l' in metric:
      rouge_l_f, rouge_l_p, rouge_l_r = rouge_quick.rouge_l_summary_level(summ, ref)
      result['rouge-l-f'] = rouge_l_f
    return result
    #return rouge_1_f, rouge_1_p, rouge_1_r, rouge_2_f, rouge_2_p, rouge_2_r, rouge_l_f, rouge_l_p, rouge_l_r


def copy_rate(summ, article):
    '''
    summ: summary that need to be calculated, a list of sentences
    article: article of the summary, a single string
    '''
    def _lcs(x, y):
        n, m = len(x), len(y)
        lcs_len = 0
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

    lcs_total = 0
    summary_len = 0
    for s in summ:
        lcs_total += _lcs(s.split(), article.split())
        summary_len += len(s.split())

    score = float(lcs_total) / summary_len if summary_len > 0 else 0.0
    return score


def reward(summ, ref, article):
    rouges = rouge_scores_quick(summ, ref)
    copy_score = copy_rate(summ, article)
    reward = rouges['rouge-1-f'] + rouges['rouge-2-f'] + rouges['rouge-l-f'] - copy_score
    return reward
    #return 0.5*rouge_1 + 0.5*rouge_2 + 0.5*rouge_l - 0.5*copy_score


if __name__ == '__main__':

    HYP="Tokyo is the one of the biggest city in the world."
    REF="The capital of Japan, Tokyo, is the center of Japanese economy."

    summ = ['a b c d', 'e f g h', 'i j k l']
    ref = ['a e h j', 'r q g h', 'i j z x c v']
    print rouge_scores(summ, ref)
    print rouge_scores_quick(summ, ref)
    pdb.set_trace()




