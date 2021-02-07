import codecs
import json
import os
import shutil
import time
from itertools import groupby
from multiprocessing import Pool
import numpy as np
import sentencepiece
from scipy.special import softmax

import pyrouge
from tools.cal_rouge import chunks
from matplotlib import pyplot as plt


def process_sequential(data):
    """
    Calculate rouge score between two sentences. Lists of sentences are evaluate sequentially.
    @param data: (candidate_list, reference_list, pool_id)
    @return: list of rouge score dictionaries
    """
    candidates, references, pool_id, debug = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    results = []

    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(1), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(1), "w",
                      encoding="utf-8") as f:
                f.write(references[i])

            r = pyrouge.Rouge155("./pyrouge/rouge/tools/ROUGE-1.5.5/")
            r.model_dir = tmp_dir + "/reference/"
            r.system_dir = tmp_dir + "/candidate/"
            r.model_filename_pattern = 'ref.#ID#.txt'
            r.system_filename_pattern = r'cand.(\d+).txt'
            rouge_results = r.convert_and_evaluate()
            if debug:
                print(rouge_results)
            results_dict = r.output_to_dict(rouge_results)
            results.append(results_dict)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results


def test_rouge(cand, ref, num_processes, debug=True):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]
    print(candidates)
    if debug:
        print('Num of candidates: %d' % len(candidates))
        print('Num of references: %d' % len(references))
    assert len(candidates) == len(references), "!!!!!!! Note: The number of candidates is " \
                                               "not equal to the number of references!!!!!!!"

    candidates_chunks = list(
        chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(
        chunks(references, int(len(references) / num_processes)))

    n_pool = len(candidates_chunks)
    print(f"Number of pools: {n_pool}")
    arg_lst = []
    for i in range(n_pool):
        print(candidates_chunks[i])
        arg_lst.append((candidates_chunks[i], references_chunks[i], i, debug))

    pool = Pool(n_pool)
    results = pool.imap(process_sequential, arg_lst)
    pool.close()
    pool.join()
    res = [r for proc in list(results) for r in proc]
    return res


def calculate_rouge(summary, paragraphs):
    """
    summary: "text..."
    paragraphs: ["para1","para2",..]
    """
    result = [test_rouge([summary], [paragraph], 1, debug=False) for paragraph in paragraphs]
    return result


def zip_data(summaries, paragraphs):
    """
    Combines each sentence of all summaries with each sentence of the corresponding input paragraphs.

    @param summaries list of summaries of form [example, sentences]
    @param paragraphs: list of input paragraphs [example, paragraph, sentences]
    @return: two lists of sentences
    """
    left_list = []
    right_list = []

    max_summary_sen = 0
    num_examples = len(summaries)
    for e, summary in enumerate(summaries):
        max_summary_sen = max(max_summary_sen, len(summary))
        for summary_sentence in summary:
            # goal: sum_sen_1 - para_1_sen_1, sum_sen_1 - para_1_sen_2 ... sum_sen_n - para_l_sen_k
            for p, paragraph in enumerate(paragraphs[e]):
                for ps, paragraph_sentence in enumerate(paragraph):
                    left_list.append(summary_sentence)
                    right_list.append(paragraph_sentence)
    return left_list, right_list, max_summary_sen, num_examples


def extract_meta(summaries, paragraphs):
    result_meta = []
    for e, summary in enumerate(summaries):
        meta = {
            "summary_sentences": len(summaries[e]),
            "paragraphs": len(paragraphs[e]),
            "paragraph_sentences": [len(para_sent) for para_sent in paragraphs[e]]
        }
        result_meta.append(meta)
    return result_meta


# result = calculate_rouge("hallo wie gehts", ["hallo ich bin doof", "autobahn", "kekse"])

def stuff(can_path, input_paragraphs, split_paragraphs=False):
    candidates = codecs.open(can_path, encoding="utf-8")
    candidates = [line.strip() for line in candidates]

    test_json = json.load(open(input_paragraphs))

    scores = []
    for i, example in enumerate(candidates):
        paragraphs = test_json[i]["src_str"]
        sentences = example.split("<q>")
        if split_paragraphs:
            paragraph_scores = []
            for paragraph in paragraphs:
                tmp = []
                for sen in paragraph.split(" ."):
                    if len(sen) > 5:
                        tmp.append(sen)
                score_example = [test_rouge(sentences, [par_sen] * len(sentences), 5, False) for par_sen in tmp]
                paragraph_scores.append(score_example)
            scores.append(paragraph_scores)
        else:
            score_example = [test_rouge(sentences, [paragraph] * len(sentences), 5, False) for paragraph in paragraphs]
            scores.append(score_example)
    return scores


# results = stuff(can_path, input_paragraphs, split_paragraphs=True)
# with open('rouge_slow.json', 'w') as f:
#    json.dump(results, f)


def tokenize(can_path="results/graphsum_multinews/test_final_preds.candidate",
             input_paragraphs="data/MultiNews_data_tfidf_paddle_paragraph_small/test/MultiNews.30.test.0.json",
             spm_path="data/spm9998_3.model"):
    """
    Uses spm model to tokenize input paragraphs and generated summaries. Results are strings (concatenated tokens).
    @param can_path: path where summaries are stored
    @param input_paragraphs: path were input paragraphs are stored
    @param spm_path: path to spm model
    @return: tuple of tokenized summaries and tokenized input paragraphs
    """
    spm = sentencepiece.SentencePieceProcessor()
    spm.load(spm_path)
    candidates = codecs.open(can_path, encoding="utf-8")
    candidates = [line.strip() for line in candidates]

    test_json = json.load(open(input_paragraphs))
    candidates_tokenized_str = [
        [' '.join(str(e) for e in spm.encode_as_ids(candidate_sen)) for candidate_sen in candidate.split("<q>")] for
        candidate in candidates]

    # candidates_str = [candidate.split("<q>") for candidate in candidates]

    """decoded = [int(e) for e in candidates_tokenized[0][0].split(' ')]

    print(spm.decode(decoded))
    print(candidates_str[0][0])"""

    documents_tokenized = [[spm.encode_as_ids(par) for par in test_json[e]['src_str']] for e in range(len(test_json))]

    # documents_tokenized_str: (#examples,#paragraph, #sentences)
    # candidates_tokenized_str: (#examples, #sentences)

    documents_tokenized_str = [
        [[' '.join(str(g) for g in (list(group) + [11])) for k, group in groupby(paragraph, lambda x: x == 11) if not k]
         for paragraph in example] for example in documents_tokenized]

    return candidates_tokenized_str, documents_tokenized_str


def transform_results(summaries, paragraphs, rouge_scores, num_examples, max_sum_sen):
    """
    Uses summaries, paragraphs and rouge results to generate numpy array. Input summaries and paragraphs are
    necessary to assign each rouge score to the right sentences.

    @param summaries: [example, sentences]
    @param paragraphs: [example, paragraph, sentences]
    @param rouge_scores: list of rouge score dictionaries
    @param num_examples: total number of examples
    @param max_sum_sen: maximal number of sentences in summary
    @return: numpy array of shape (num_examples, num_paragrapgs, num_paragraph_sen, max_sum_sen, 3)
             where the last dimension contains ROUGE-1, ROUGE-2 and ROUGE-L
    """
    num_paragraphs = 30
    num_paragraph_sen = 30

    res = -np.ones((num_examples, num_paragraphs, num_paragraph_sen, max_sum_sen, 3))
    cnt = 0
    for e, summary in enumerate(summaries):
        for ss, summary_sentence in enumerate(summary):
            for p, paragraph in enumerate(paragraphs[e]):
                for ps, _ in enumerate(paragraph):
                    res[e, p, ps, ss, 0] = rouge_scores[cnt]["rouge_1_f_score"]
                    res[e, p, ps, ss, 1] = rouge_scores[cnt]["rouge_2_f_score"]
                    res[e, p, ps, ss, 2] = rouge_scores[cnt]["rouge_l_f_score"]
                    cnt += 1
    return res


def extract_rouge(tokenized_summaries, tokenized_paragraphs):
    """
    Generates ROUGE scores between each sentence of each summary and each sentence of the corresponding paragraphs.

    @param tokenized_summaries:
    @param tokenized_paragraphs:
    @return: ROUGE scores (numpy array)
    """
    (left, right, max_sum_sen, num_examples) = zip_data(tokenized_summaries, tokenized_paragraphs)
    results = test_rouge(left, right, 12, False)
    results = transform_results(tokenized_summaries, tokenized_paragraphs, results, num_examples, max_sum_sen)
    return results


# with open('rouge.p', 'wb') as fp:
#    pickle.dump(results, fp)

def slice_rouge(scores, example, rouge_meta, stack=True):
    """
    Remove padding in rouge score array for a given example.
    @param scores: rouge scores (padded)
    @param example: example to process
    @param rouge_meta: meta data from rouge processing
    @param stack: stack all paragraph sentences (or return list for each paragraph)
    @return: rouge scores of size (total_paragraph_sentences, summary_sentences) (stacked)
             or [(paragraph_sentences), summary_sentences)] for each paragraph
    """
    scores_example = scores[example]
    patches = [scores_example[para, :rouge_meta[example]["paragraph_sentences"][para],
               :rouge_meta[example]["summary_sentences"]]
               for para in range(rouge_meta[example]["paragraphs"])]
    if stack:
        img = np.vstack(patches)
        return img
    return patches


def plot_rouge(scores, example, input_meta, rouge_meta, score=1, cmap="hot"):
    """
    Plot rouge scores between paragraph sentences and summary sentences (for a single example) as heatmap.
    Document boundaries and paragraph boundaries are marked as vertical lines.
    @param scores: rouge scores
    @param example: example to plot
    @param input_meta: meta data from preprocessing script (number of textual units)
    @param rouge_meta: meta data from rouge processing (number of sentences per paragraph)
    @param score: plot ROUGE-1, ROUGE-2 or ROUGE-L
    @param cmap: color map for plot
    """

    rouge_dict = {0: "1", 1: "2", 2: "L"}
    fig, ax = plt.subplots(figsize=(20, 10))

    img = slice_rouge(scores, example, rouge_meta)

    im = ax.imshow(img[:, :, score].T, cmap=cmap, interpolation='nearest')
    sum_sen = 0
    ax.set_xlabel("input sentence")
    ax.set_ylabel("summary sentence")
    for para in range(rouge_meta[example]["paragraphs"]):
        sum_sen = sum_sen + rouge_meta[example]["paragraph_sentences"][para]
        ax.axvline(sum_sen - 0.5, color="cyan", linewidth=1)

    number_of_textual_units = input_meta["number_of_textual_units"][example]

    text_units = np.cumsum(
        number_of_textual_units[number_of_textual_units != 0])

    for x in text_units[:-1]:
        ax.axvline(sum(rouge_meta[example]["paragraph_sentences"][:x]) - 0.5, color="w", linewidth=3)

    fig.colorbar(im, ax=ax, label=f"ROUGE {rouge_dict[score]}")
    plt.show()


def aggregate_rouge_paragraph(example, rouge_scores, rouge_meta, fun=np.mean):
    """
    Aggregate rouge scores for each paragraph.
    @param example: example to process
    @param rouge_scores: rouge score array
    @param rouge_meta: meta information obtained by rouge processing
    @param fun: aggregate function (np.mean, np.max etc.)
    @return: aggregated rouge score (num_paragraphs, num_summary_sentences)
    """
    rouge_scores_sentences = slice_rouge(rouge_scores, example, rouge_meta, stack=False)
    aggregated = [fun(rouge_scores_sentences[para], axis=0, keepdims=True) for para in
                  range(rouge_meta[example]["paragraphs"])]
    return np.vstack(aggregated)


def plot_attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, attention_dec=7, attention_mh=7,
                                     attention_metric="Mean", aggregate_function=np.mean):
    """
    Plot correlation between rouge scores and attention weights over all examples.
    @param rouge_scores: rouge scores
    @param rouge_meta: meta information obtained by ROUGE processing
    @param attention_weights: attention weights
    @param attention_dec: decoder layer
    @param attention_mh: multi-head attention layer
    @param attention_metric: metric used for attention aggregation ("Mean", "Median",...)
    @param aggregate_function: aggregation function for ROUGE aggregation (np.mean, ...)
    """
    corr, r1, r2, rl, attentions = attention_rouge_correlation(
        rouge_scores, rouge_meta, attention_weights, attention_dec=attention_dec,
        attention_mh=attention_mh, attention_metric=attention_metric, aggregate_function=aggregate_function)
    plt.figure(figsize=(20, 10))
    plt.scatter(r1, attentions, label="ROUGE 1")
    plt.scatter(r2, attentions, label="ROUGE 2")
    plt.scatter(rl, attentions, label="ROUGE L")
    plt.xlabel("ROGUE")
    plt.ylabel("Attention")
    plt.legend()


def plot_attention_rouge_correlation_all_examples(rouge_scores, rouge_meta, attention_weights, attention_dec=7,
                                                  attention_mh=7, attention_metric="Mean", aggregate_function=np.mean):
    """
    Plot correlation between rouge scores and attention weights for each example.
    @param rouge_scores: rouge scores
    @param rouge_meta: meta information obtained by ROUGE processing
    @param attention_weights: attention weights
    @param attention_dec: decoder layer
    @param attention_mh: multi-head attention layer
    @param attention_metric: metric used for attention aggregation ("Mean", "Median",...)
    @param aggregate_function: aggregation function for ROUGE aggregation (np.mean, ...)
    """
    fig, axes = plt.subplots(10, 3)
    fig.set_size_inches(20, 20)

    for i, s in enumerate(["ROUGE 1", "ROUGE 2", "ROUGE L"]):
        axes[0, i].set_title(s)
    for e in range(10):
        r_tmp = np.transpose(aggregate_rouge_paragraph(e, rouge_scores, rouge_meta, fun=aggregate_function),
                             axes=(1, 0, 2))
        a_tmp = attention_weights[attention_metric][e, 0, :np.shape(r_tmp)[0], attention_dec, attention_mh,
                :np.shape(r_tmp)[1]]

        r_tmp = softmax(r_tmp, axis=1)
        r_tmp = r_tmp.reshape((-1, 3))

        a_tmp = a_tmp.reshape(-1, )
        r1 = r_tmp[:, 0]
        r2 = r_tmp[:, 1]
        rl = r_tmp[:, 2]

        #corr = np.corrcoef(np.array([a_tmp, r1, r2, rl]))

        axes[e, 0].scatter(r1, a_tmp)
        axes[e, 1].scatter(r2, a_tmp)
        axes[e, 2].scatter(rl, a_tmp)


def attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, attention_dec=7, attention_mh=7,
                                attention_metric="Mean", aggregate_function=np.mean):
    """
    Calculates correlation between ROUGE scores and attention weights over all examples.
    ROUGE score is normalized by using softmax function.

    @param rouge_scores: rouge scores
    @param rouge_meta: meta information obtained by ROUGE processing
    @param attention_weights: attention weights
    @param attention_dec: decoder layer
    @param attention_mh: multi-head attention layer
    @param attention_metric: metric used for attention aggregation ("Mean", "Median",...)
    @param aggregate_function: aggregation function for ROUGE aggregation (np.mean, ...)
    @return: correlation_matrix, processed_r1, processed_r2, processed_rl, processed_attention
    """
    num_examples = len(rouge_meta)
    rouges = -np.ones((num_examples, 30 * 12, 3))
    attentions = -np.ones((num_examples, 30 * 12))

    for e in range(num_examples):
        r_tmp = np.transpose(aggregate_rouge_paragraph(e, rouge_scores, rouge_meta, fun=aggregate_function),
                             axes=(1, 0, 2))
        a_tmp = attention_weights[attention_metric][e, 0, :np.shape(r_tmp)[0], attention_dec, attention_mh,
                :np.shape(r_tmp)[1]]

        # r1 = r1 / np.sum(r1, axis=0)

        r_tmp = softmax(r_tmp, axis=1)
        r_tmp = r_tmp.reshape((-1, 3))

        rouges[e, :r_tmp.shape[0], :r_tmp.shape[1]] = r_tmp
        a_tmp = a_tmp.reshape((-1,))
        attentions[e, :a_tmp.shape[0]] = a_tmp

    rouges = rouges.reshape((-1, 3))
    r1 = rouges[:, 0]
    r1 = r1[r1 != -1]

    r2 = rouges[:, 1]
    r2 = r2[r2 != -1]

    rl = rouges[:, 2]
    rl = rl[rl != -1]

    attentions = attentions.reshape(-1, )
    attentions = attentions[attentions != -1]
    corr = np.corrcoef([attentions, r1, r2, rl])
    return corr, r1, r2, rl, attentions


if __name__ == "__main__":
    can_path = "results/graphsum_multinews/test_final_preds.candidate"
    input_paragraphs = "data/MultiNews_data_tfidf_paddle_paragraph_small/test/MultiNews.30.test.0.json"

    summaries, paragraphs = tokenize(can_path=can_path, input_paragraphs=input_paragraphs)
    #  results = extract_rouge(summaries, paragraphs)
    # np.save("rouge_full", results)
    meta = extract_meta(summaries, paragraphs)
    json.dump(meta, open("rouge_meta.json", "w"))
