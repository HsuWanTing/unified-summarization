import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pk
import os, sys
import itertools
import util
import time
import rouge_not_a_wrapper as my_rouge
import pdb
from tqdm import tqdm


def plot_gt_sent_num(x, y, result_dir):
  plt.figure(figsize=(9, 5))
  plt.plot(x, y, '-bo')
  plt.xticks(np.arange(1, 31, 1))
  plt.grid(True)
  plt.xlabel("Number of Sentences")
  plt.ylabel("Number of Articles")
  plt.title("Number of Ground-Truth Selected Sentences")
  plt.savefig(os.path.join(result_dir, 'gt_select_num.png'))
  plt.clf()

def plot_art_sent_num(art_sent_num, result_dir):
  plt.figure(figsize=(9, 5))
  plt.hist(np.array(art_sent_num), bins=np.arange(0, 111, 5))
  plt.xticks(np.arange(0, 111, 5))
  plt.grid(True)
  plt.xlabel("Number of Sentences")
  plt.ylabel("Number of Articles")
  plt.title("Number of Ground-Truth Selected Sentences")
  plt.savefig(os.path.join(result_dir, 'art_sent_num.png'))
  plt.clf()

def plot_relation(x, y, result_dir):
  plt.figure(figsize=(9, 5))
  plt.plot(x, y, '-ro')
  plt.xticks(np.arange(0, 31, 1))
  plt.xlabel("Number of GT Selected Sentences")
  plt.ylabel("Average Number of Article Sentences")
  plt.grid(True)
  plt.savefig(os.path.join(result_dir, 'relation.png'))
  plt.clf()

# Change color of each axis
def color_y_axis(ax, color):
  """Color your axes."""
  for t in ax.get_yticklabels():
    t.set_color(color)

def plot_gt_hist_relation(x, y1, y2, result_dir):
  fig, ax1 = plt.subplots(figsize=(9, 5))
  plt.xticks(np.arange(0, 31, 1))
  ax2 = ax1.twinx()

  ax1.plot(x, y1, '-bo')
  ax1.grid(True)
  ax1.set_xlabel('Number of GT Selected Sentences')
  ax1.set_ylabel('Number of Articles')

  ax2.plot(x, y2, '-ro')
  ax2.set_ylabel('Average Number of Article Sentences')

  color_y_axis(ax1, 'b')
  color_y_axis(ax2, 'r')
  plt.savefig(os.path.join(result_dir, 'gt_sent_num_relation.png'))
  plt.clf()
  print '[Info] plot GT and article sent num relation'

def plot_PRC_RRC(articles, gt_ids, probs, step, method, result_dir):
  sent_nums, precisions, recalls, accuracys, ratios, avg_ps, avg_rs, avg_as = util.get_batch_precision_recall(articles, gt_ids, probs, step, method, tf_print=False)
  plt.figure(figsize=(9, 9))
  plt.plot(recalls, precisions, '-ro')
  plt.xticks(np.arange(0.0, 1.001, 0.05))
  plt.yticks(np.arange(0.0, 1.001, 0.05))
  plt.grid(True)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision Recall Curve (use " + method + " as threshold)")
  plt.savefig(os.path.join(result_dir, 'prc_' + method + '.png'))
  plt.clf()
  print '[Info] plot PRC'

  plt.figure(figsize=(9, 9))
  plt.plot(recalls, ratios, '-ro')
  plt.xticks(np.arange(0.0, 1.001, 0.05))
  plt.yticks(np.arange(0.0, 1.001, 0.05))
  plt.grid(True)
  plt.xlabel("Recall")
  plt.ylabel("Select Ratio")
  plt.title("Select Ratio Recall Curve (use " + method + " as threshold)")
  plt.savefig(os.path.join(result_dir, 'rrc_' + method + '.png'))
  plt.clf()
  print '[Info] plot RRC'

  fig, ax1 = plt.subplots(figsize=(9, 9))
  bins = np.arange(0.0, 1.0, step)
  plt.xticks(np.arange(0.0, 1.001, 0.05))
  plt.title("Accuracy and Selected Number (use " + method + " as threshold)")
  ax2 = ax1.twinx()

  ax1.plot(bins, accuracys, '-ro')
  ax1.grid(True)
  ax1.set_xlabel('Threshold')
  ax1.set_ylabel('Accuracy')

  ax2.plot(bins, sent_nums, '-bo')
  ax2.set_ylabel("Number of selected sentences")

  color_y_axis(ax1, 'r')
  color_y_axis(ax2, 'b')
  plt.savefig(os.path.join(result_dir, 'acc_' + method + '.png'))
  plt.clf()
  print '[Info] plot Accuracy'

def plot_select_delete_scores(articles, gt_ids, probs, scores, step, method, result_dir):
  thresholds = np.arange(0.0, 1.0, step)[1:]
  for thres in thresholds:
    select_scores = []
    delete_scores = []
    precisions = []
    recalls = []
    ratios = []
    for i in range(len(articles)):
      _, selected_ids, precision, recall, accuracy, ratio = util.get_select_accuracy_one_thres(\
                                                                  articles[i], probs[i], gt_ids[i], \
                                                                  thres, method=method)
      select_scores += [scores[i][idx] for idx in selected_ids]
      delete_scores += [scores[i][idx] for idx in range(len(articles[i])) if idx not in selected_ids]
      precisions.append(precision)
      recalls.append(recall)
      ratios.append(ratio)

    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)
    ratio = sum(ratios)/len(ratios)

    plt.figure(figsize=(7, 5))
    d_scores, _, _ = plt.hist(np.array(delete_scores), bins=np.arange(0.0, 0.71, 0.05), label='deleted sentences')
    plt.hist(np.array(select_scores), bins=np.arange(0.0, 0.71, 0.05), bottom=d_scores, label='selected sentences')
    plt.xticks(np.arange(0.0, 0.71, 0.05))
    plt.grid(True)
    plt.xlabel("rouge score")
    plt.ylabel("number of sentences")
    plt.title("Selected/Deleted Scores (thres: %.2f, recall: %.3f, ratio: %3f)" % (thres, recall, ratio))
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'select_delete_scores_%.2f.png' %(thres)))
    plt.clf()
    print '[Info] plot select delete scores (thres: ', thres, ')'

def plot_gt_scores(gt_ids, scores, result_dir):
  select_scores = []
  delete_scores = []
  for i in range(len(scores)):
    select_scores += [scores[i][idx] for idx in gt_ids[i]]
    delete_scores += [scores[i][idx] for idx in range(len(scores[i])) if idx not in gt_ids[i]]

  plt.figure(figsize=(7, 5))
  d_scores, _, _ = plt.hist(np.array(delete_scores), bins=np.arange(0.0, 0.71, 0.05), label='deleted sentences')
  plt.hist(np.array(select_scores), bins=np.arange(0.0, 0.71, 0.05), bottom=d_scores, label='selected sentences')
  plt.xticks(np.arange(0.0, 0.71, 0.05))
  plt.grid(True)
  plt.xlabel("rouge score")
  plt.ylabel("number of sentences")
  plt.title("GT Selected/Deleted Scores")
  plt.legend()
  plt.savefig(os.path.join(result_dir, 'gt_select_delete_scores.png'))
  plt.clf()
  print '[Info] plot GT scores'

def plot_rougeL_r(select_sents, reference_sents, result_dir):
  Rouge_l_rs = []
  for i in tqdm(range(len(select_sents))):
    _, _, Rouge_l_r = my_rouge.rouge_l_summary_level(select_sents[i], reference_sents[i])
    Rouge_l_rs.append(Rouge_l_r)

  plt.figure(figsize=(9, 5))
  plt.hist(np.array(Rouge_l_rs), bins=np.arange(0.0, 1.01, 0.05))
  plt.xticks(np.arange(0.0, 1.01, 0.05))
  plt.grid(True)
  plt.xlabel("Rouge-L recall")
  plt.ylabel("number of articles")
  plt.title("Rouge-L Recall of Ground-Truth Selected Sentences")
  plt.savefig(os.path.join(result_dir, 'gt_rougeL_r.png'))
  plt.clf()


def plot_all(result_dir):
  # load data
  if 'test' in result_dir:
    scores_data = pk.load(open('/data/VSLab/cindy/Workspace/summarization/data/CNN_Dailymail/Rouge_scores/test_scores2.pkl'))
  elif 'val' in result_dir:
    scores_data = pk.load(open('/data/VSLab/cindy/Workspace/summarization/data/CNN_Dailymail/Rouge_scores/val_scores2.pkl'))
  elif 'train' in result_dir:
    scores_data = pk.load(open('/data/VSLab/cindy/Workspace/summarization/data/CNN_Dailymail/Rouge_scores/train_scores2.pkl'))
  scores = scores_data['rougeL']['r']

  data = pk.load(open(os.path.join(result_dir, 'probs.pkl')))
  gt_sent_num = [len(data['gt_ids'][i]) for i in data['article']]
  art_sent_num = [len(data['article'][i]) for i in data['article']]
  articles = scores_data['article']
  references = [data['reference'][i] for i in data['article']]
  gt_ids = [data['gt_ids'][i] for i in data['article']]
  gt_sents = [[articles[i][idx] for idx in data['gt_ids'][i]] for i in data['article']]
  probs = [data['probs'][i] for i in data['article']]
  scores = [scores[i] for i in range(len(scores)) if len(scores_data['article'][i]) > 0]

  # get gt sent num histogram
  hist, _, _ = plt.hist(np.array(gt_sent_num), bins=np.arange(0, 32, 1))
  plt.clf()

  # get relation
  avg_art_num = []
  bins = np.arange(1, 31, 1)
  #have_data_bins = []
  for i in range(len(bins)):
    art_len = []
    for j in data['gt_ids']:
      if len(data['gt_ids'][j]) == bins[i]:
        art_len.append(len(data['article'][j]))
    if len(art_len) > 0:
      avg_art_num.append(float(sum(art_len))/len(art_len))
      #have_data_bins.append(bins[i])
    else:
      avg_art_num.append(0)

  #plot_rougeL_r(gt_sents, references, result_dir)
  #plot_gt_scores(articles, gt_ids, scores, result_dir)
  plot_gt_sent_num(bins, hist[1:], result_dir)
  plot_art_sent_num(art_sent_num, result_dir)
  plot_relation(bins, avg_art_num, result_dir)
  plot_gt_hist_relation(bins, hist[1:], avg_art_num, result_dir)
  plot_PRC_RRC(articles, gt_ids, probs, 0.01, 'prob', result_dir)
  plot_PRC_RRC(articles, gt_ids, probs, 0.01, 'ratio', result_dir)
  plot_select_delete_scores(articles, gt_ids, probs, scores, 0.1, 'prob', result_dir)

def compare_2_model(result_dir1, result_dir2, name1, name2, step, method):
  # load data
  data1 = pk.load(open(os.path.join(result_dir1, 'probs.pkl')))
  articles1 = [data1['article'][i] for i in data1['article']]
  gt_ids1 = [data1['gt_ids'][i] for i in data1['article']]
  probs1 = [data1['probs'][i] for i in data1['article']]

  # load data
  data2 = pk.load(open(os.path.join(result_dir2, 'probs.pkl')))
  articles2 = [data2['article'][i] for i in data2['article']]
  gt_ids2 = [data2['gt_ids'][i] for i in data2['article']]
  probs2 = [data2['probs'][i] for i in data2['article']]

  sent_nums1, precisions1, recalls1, _, ratios1, avg_ps1, avg_rs1, _ = util.get_batch_precision_recall(articles1, gt_ids1, probs1, step, method, tf_print=False)
  sent_nums2, precisions2, recalls2, _, ratios2, avg_ps2, avg_rs2, _ = util.get_batch_precision_recall(articles2, gt_ids2, probs2, step, method, tf_print=False)
  plt.figure(figsize=(9, 9))
  l1, = plt.plot(recalls1, precisions1, '-ro')
  l2, = plt.plot(recalls1, precisions2, '-bo')
  plt.legend([l1, l2], [name1, name2])
  plt.xticks(np.arange(0.0, 1.001, 0.05))
  plt.yticks(np.arange(0.0, 1.001, 0.05))
  plt.grid(True)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision Recall Curve (use " + method + " as threshold)")
  plt.savefig(os.path.join(result_dir1, 'prc_' + method + '_' + name1 + '_' + name2 + '.png'))
  plt.clf()

  plt.figure(figsize=(9, 9))
  plt.plot(recalls1, ratios1, '-ro')
  plt.plot(recalls2, ratios2, '-bo')
  plt.legend([l1, l2], [name1, name2])
  plt.xticks(np.arange(0.0, 1.001, 0.05))
  plt.yticks(np.arange(0.0, 1.001, 0.05))
  plt.grid(True)
  plt.xlabel("Recall")
  plt.ylabel("Select Ratio")
  plt.title("Select Ratio Recall Curve (use " + method + " as threshold)")
  plt.savefig(os.path.join(result_dir1, 'rrc_' + method + '_' + name1 + '_' + name2 + '.png'))
  plt.clf()


if __name__=='__main__':

  t1 = time.time()
  if len(sys.argv) == 2:
    plot_all(sys.argv[1])
  elif len(sys.argv) == 5:
    compare_2_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 0.01, 'prob')
  t2 = time.time()
  print '[Info] analysis time: ', (t2-t1)/60.0, 'mins'
