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
import pickle
from scipy.special import softmax

import sys
sys.path.insert(0, './src')

import pyrouge
from tools.cal_rouge import chunks
from matplotlib import pyplot as plt


def plot_rouge(scores, input_meta, rouge_meta, score=1, cmap="hot"):
    """
    Plot rouge scores between paragraph sentences and summary sentences (for a single example) as heatmap.
    Document boundaries and paragraph boundaries are marked as vertical lines.
    @param scores: rouge scores
    @param input_meta: meta data from preprocessing script (number of textual units)
    @param rouge_meta: meta data from rouge processing (number of sentences per paragraph)
    @param score: plot ROUGE-1, ROUGE-2 or ROUGE-L
    @param cmap: color map for plot
    """

    rouge_dict = {0: "1", 1: "2", 2: "L"}
    #fig, ax = plt.subplots(figsize=(20, 10))
    
    sentence_information = np.zeros(shape=[100])
    
    histo = np.zeros((30, 100))
    
    for ex in range(500):

        img = slice_rouge(scores, ex, rouge_meta)




        number_of_paragraphs_each_document = np.cumsum([0] + list(input_meta["number_of_textual_units"][ex]))

        sentence_cumsum = [0] + list(np.cumsum(rouge_meta[ex]["paragraph_sentences"]))

        number_of_documents = np.where(number_of_paragraphs_each_document == rouge_meta[ex]["paragraphs"])[0][1]



 

        for i in range(number_of_documents-1):

            document_start = number_of_paragraphs_each_document[i]
            document_ending = number_of_paragraphs_each_document[i+1]

            #print(f"document_start: {document_start} | document_ending: {document_ending}")
            sentence_start = sentence_cumsum[document_start]
            sentence_ending = sentence_cumsum[document_ending]
            
            sentence_information[sentence_ending-sentence_start] += 1

            #print(f"sentence_start: {sentence_start} | sentence_ending: {sentence_ending}")
            relevent_scores = img[sentence_start:sentence_ending,:,score].T

            for j, x in enumerate(np.argmax(relevent_scores, axis=1)):
                    histo[j, x] += 1
        
    histo = histo / (np.cumsum(sentence_information[::-1])[::-1] + np.e**-15)
    
    
    histo = histo/(histo.sum(axis=1) + np.e**-15)[:, np.newaxis]
    
    plt.imshow(histo[:20,:20].T)
    plt.xlabel("Generated Sentences")
    plt.ylabel("Input Sentences")
    plt.colorbar()

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
                  range(rouge_meta[example]["paragraphs"]) if len(rouge_scores_sentences[para]) > 0]
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
    print(corr[0,:])
    
    plt.figure(figsize=(20, 10))
    plt.title(f"Correlation to ROUGE-F(1/2/L): {corr[0,1]:.2f} / {corr[0,2]:.2f} / {corr[0,3]:.2f}")
    plt.suptitle("Visualizatzion of Correlation between ROUGE-Scores and the mean Attention-Weights of Decoding Layers %s" % str(np.array(attention_dec)+1))
    plt.scatter(r1, attentions, label="ROUGE 1")
    plt.scatter(r2, attentions, label="ROUGE 2")
    plt.scatter(rl, attentions, label="ROUGE L")
    plt.xlabel("ROGUE-Scores")
    plt.ylabel("Global Attention Weights")
    plt.savefig("correlation_rouge_attention_%s.png" % str(np.array(attention_dec)+1))
    plt.legend()




def attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, attention_dec=[7],
                                attention_metric="Mean", aggregate_function=np.mean):
    """
    Calculates correlation between ROUGE scores and attention weights over all examples.
    ROUGE score is normalized by using softmax function.

    @param rouge_scores: rouge scores
    @param rouge_meta: meta information obtained by ROUGE processing
    @param attention_weights: attention weights
    @param attention_dec: decoder layer
    @param attention_metric: metric used for attention aggregation ("Mean", "Median",...)
    @param aggregate_function: aggregation function for ROUGE aggregation (np.mean, ...)
    @return: correlation_matrix, processed_r1, processed_r2, processed_rl, processed_attention
    """
    
    num_examples,_,num_generated_sentences,_,_,num_paragraphs = attention_weights[attention_metric].shape
        
    rouges = -np.ones((num_examples, num_paragraphs * num_generated_sentences, 3))
    attentions = -np.ones((num_examples, num_paragraphs * num_generated_sentences))
    
    for e in range(num_examples):
        r_tmp = np.transpose(aggregate_rouge_paragraph(e, rouge_scores, rouge_meta, fun=aggregate_function),
                             axes=(1, 0, 2))
        a_tmp = attention_weights[attention_metric][e, 0, :np.shape(r_tmp)[0], attention_dec, :,
                :np.shape(r_tmp)[1]]
        

        a_tmp = np.mean(a_tmp, axis=0)
        a_tmp = np.mean(a_tmp, axis=1)
        
        
           
        
        #r_tmp = r_tmp / (np.sum(r_tmp, axis=1, keepdims=True) + np.e**-15)

    
        #r_tmp = softmax(r_tmp, axis=1)
          
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
    
    """index = np.where(attentions > 0.1)
    
    attentions = attentions[index]
    r1 = r1[index]
    r2 = r2[index]
    rl = rl[index]"""
   
    
    corr = np.corrcoef([attentions, r1, r2, rl])
    return corr[0,:], r1, r2, rl, attentions


if __name__ == "__main__":
         
    parser = argparse.ArgumentParser(description='Calculate Rouge Score for similarity measurement for source origin analysis')
    
    parser.add_argument("--rouge_information_path", default='rouge_information/')
    parser.add_argument("--transformed_attention_weights_path", default='transformed_attention_weights')
    parser.add_argument("--aggregation_metric", default='Mean')
    parser.add_argument("--aggregate_function", default='np.median')
    parser.add_argument("--result_output", default='correlation_results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_output):
        os.mkdir(args.result_output)
        
    rouge_scores = np.load(os.path.join(args.rouge_information_path,"rouge_full.npy"))
    rouge_meta = json.load(open(os.path.join(args.rouge_information_path,"rouge_meta.json")))
    
    attention_weights = pickle.load(open(os.path.join(args.transformed_attention_weights_path,'sentence_level_aggregation_dict'),'rb'))
    
    aggregate_function = eval(args.aggregate_function)
        
    num_decoding_layers = attention_weights[args.aggregation_metric].shape[3]
                                    
    corr_matrix = np.zeros(shape=[num_decoding_layers, 4])                                
    
    r1_list = {"all": [], "first": [], "last": []}
    r2_list = {"all": [], "first": [], "last": []}                                
    rl_list = {"all": [], "first": [], "last": []}
    attention_list = {"all": [], "first": [], "last": []}
                                    
    
    for idx in range(num_decoding_layers):
        corr, r1, r2, rl, attentions = attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, [idx], args.aggregation_metric, aggregate_function)
        
        print(corr)
        corr_matrix[idx,:] = corr
        r1_list["all"].extend(r1)
        r2_list["all"].extend(r2)
        rl_list["all"].extend(rl)                    
        attention_list["all"].extend(attentions)                            
    
    print(f"Correlation Matrix: {corr_matrix}")
                                    
    first_layers = np.arange(0,num_decoding_layers/2,1).astype("int") + 1
    corr, r1, r2, rl, attentions = attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, first_layers, args.aggregation_metric, aggregate_function)
        
    first_layers_correlation = corr
    r1_list["first"].extend(r1)
    r2_list["first"].extend(r2)
    rl_list["first"].extend(rl)                    
    attention_list["first"].extend(attentions)   
    
    print(f"Correlation Matrix over first layers: {first_layers_correlation}")
                                    
    last_layers = np.arange(num_decoding_layers/2,num_decoding_layers,1).astype("int")
    corr, r1, r2, rl, attentions = attention_rouge_correlation(rouge_scores, rouge_meta, attention_weights, last_layers, args.aggregation_metric, aggregate_function)
        
    last_layers_correlation = corr
    r1_list["last"].extend(r1)
    r2_list["last"].extend(r2)
    rl_list["last"].extend(rl)                    
    attention_list["last"].extend(attentions)  
    
    print(f"Correlation Matrix over last layers: {last_layers_correlation}")

    np.save(os.path.join(args.result_output, "corr_matrix.npy"), corr_matrix)
    np.save(os.path.join(args.result_output, "first_layers_correlation.npy"), first_layers_correlation)
    np.save(os.path.join(args.result_output, "last_layers_correlation.npy"), last_layers_correlation)
                                    
    pickle.dump(r1_list, file=open(os.path.join(args.result_output, "r1_list"), "wb"))
    pickle.dump(r2_list, file=open(os.path.join(args.result_output, "r2_list"), "wb"))
    pickle.dump(rl_list, file=open(os.path.join(args.result_output, "rl_list"), "wb"))
    pickle.dump(attention_list, file=open(os.path.join(args.result_output, "attention_list"), "wb"))                                
