import codecs
import json
import os
import shutil
import time
from itertools import groupby
from multiprocessing import Pool

import numpy as np
import sentencepiece

import pyrouge
from tools.cal_rouge import chunks


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

if __name__ == "__main__":
    can_path = "results/graphsum_multinews/test_final_preds.candidate"
    input_paragraphs = "data/MultiNews_data_tfidf_paddle_paragraph_small/test/MultiNews.30.test.0.json"

    summaries, paragraphs = tokenize(can_path=can_path, input_paragraphs=input_paragraphs)
  #  results = extract_rouge(summaries, paragraphs)
   # np.save("rouge_full", results)
    meta = extract_meta(summaries, paragraphs)
    json.dump(meta, open("rouge_meta.json", "w"))
