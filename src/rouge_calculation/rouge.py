import codecs
import json
import os
import shutil
import time
from itertools import groupby
from multiprocessing import Pool
import numpy as np
import argparse
import sentencepiece
from scipy.special import softmax

import sys
sys.path.insert(0, './src')

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

            r = pyrouge.Rouge155("./pyrogue/tools/ROUGE-1.5.5")
            r.model_dir = tmp_dir + "/reference/"
            r.system_dir = tmp_dir + "/candidate/"
            r.model_filename_pattern = 'ref.#ID#.txt'
            r.system_filename_pattern = r'cand.(\d+).txt'
            rouge_results = r.convert_and_evaluate()
            #if debug:
                #print(rouge_results)
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
    #print(candidates)
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


def tokenize(can_path="results/graphsum_multinews/test_final_preds.candidate",
             input_data="data/MultiNews_data_tfidf_paddle_paragraph_small_test",
             spm_path="data/spm9998_3.model"):
    """
    Uses spm model to tokenize input paragraphs and generated summaries. Results are strings (concatenated tokens).
    @param can_path: path where summaries are stored
    @param input_data: path were input paragraphs are stored
    @param spm_path: path to spm model
    @return: tuple of tokenized summaries and tokenized input paragraphs
    """
    spm = sentencepiece.SentencePieceProcessor()
    spm.load(spm_path)
    candidates = codecs.open(can_path, encoding="utf-8")
    candidates = [line.strip() for line in candidates]

    json_files = [pos_json for pos_json in os.listdir(input_data) if pos_json.endswith('.json')]
    test_json = [ex for json_file in json_files for ex in json.load(open(os.path.join(input_data,json_file)))]
    
    candidates_tokenized_str = [
        [' '.join(str(e) for e in spm.encode_as_ids(candidate_sen)) for candidate_sen in candidate.split("<q>")] for
        candidate in candidates]

    # candidates_str = [candidate.split("<q>") for candidate in candidates]

    """decoded = [int(e) for e in candidates_tokenized[0][0].split(' ')]

    print(spm.decode(decoded))
    print(candidates_str[0][0])"""
    
    documents_tokenized = [test_json[e]['src'] for e in range(len(test_json))]

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
    num_paragraphs = np.max([
        len(ex) for ex in paragraphs])
    
    num_paragraph_sen = np.max([
        len(para) for ex in paragraphs for para in ex
    ])

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
    os.environ["PERL5LIB"] = "/home/lochner/miniconda3/envs/graph36_final/lib/site_perl/5.26.2/x86_64-linux-thread-multi:/home/lochner/miniconda3/envs/graph36_final/lib/site_perl/5.26.2"
    
    parser = argparse.ArgumentParser(description='Calculate Rouge Score for similarity measurement for source origin analysis')
    
    parser.add_argument("--can_path", default='./results/graphsum_multinews/test_final_preds.candidate')
    parser.add_argument("--input_data", default='./data/MultiNews_data_tfidf_30_paddle_full_paragraph/small_test/')
    parser.add_argument("--output_dir", default='rouge_information/')
    parser.add_argument("--spm_path", default='data/spm9998_3.model/')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    summaries, paragraphs = tokenize(can_path=args.can_path, input_data=args.input_data)
    results = extract_rouge(summaries, paragraphs)
    np.save(os.path.join(args.output_dir, "rouge_full"), results)
    meta = extract_meta(summaries, paragraphs)
    json.dump(meta, open(os.path.join(args.output_dir,"rouge_meta.json"), "w"))
