import numpy as np
import pickle
import os
import shutil
import argparse
from glob import glob
from transform_attention_weights_lib import transform_attention_weights
from aggregation_sentence_information_lib import aggregate_weight_information_for_sentences

def main():
    parser = argparse.ArgumentParser(description='Transform Global Attention Weights')
    parser.add_argument("--input_path", default='saved_attention_weights/')
    parser.add_argument("--cleanup_data", default=False, type=bool)
    parser.add_argument("--output_path", default='transformed_attention_weights/')
    parser.add_argument("--max_beam_length", default=300, type=int)
    parser.add_argument("--only_highest_beam", default=True, type=bool)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    cleaned_weight_matrix, cleaned_score_matrix, result_dict = load_data(args.input_path, args.max_beam_length, args.only_highest_beam, args.cleanup_data)
    
    
    sentence_level_aggregation = aggregate_weight_information_for_sentences(cleaned_weight_matrix, result_dict)
       
    pickle.dump(result_dict, file=open(os.path.join(args.output_path, "result_dict"), "wb"))
    pickle.dump(sentence_level_aggregation, file=open(os.path.join(args.output_path, "sentence_level_aggregation_dict"), "wb"))
    
    np.save(os.path.join(args.output_path, "cleaned_weight_matrix.npy"), cleaned_weight_matrix)
    np.save(os.path.join(args.output_path, "cleaned_score_matrix.npy"), cleaned_score_matrix)
    
    
    
def load_data(input_path, max_beam_length, only_highest_beam, cleanup_data):
    
    batch_directories = glob("%s/*/" % (input_path))
    
    batch_directories.sort(key=os.path.getmtime)
    
    weights_list = []
    scores_list = []
    result_dicts_list = []
    
    for batch in batch_directories:
        
        weights = np.load(os.path.join(batch, "pretrained_attention_weights.npy"))
        parent_idx = np.load(os.path.join(batch,"parent_idx.npy"))
        scores = np.load(os.path.join(batch, "scores.npy"))
        result_dicts = pickle.load(open(os.path.join(batch, "save_dict"),"rb"))
        
        if cleanup_data:
            shutil.rmtree(batch, ignore_errors=True)
        
        result_dicts["beam_length"] = np.where(result_dicts["beam_length"] < max_beam_length+1, result_dicts["beam_length"], max_beam_length)
        result_dicts["longest_beam_array"] = np.where(result_dicts["longest_beam_array"] < max_beam_length+1, result_dicts["longest_beam_array"], max_beam_length)
        
        
        cleaned_weight_matrix, cleaned_score_matrix = transform_attention_weights(weights, parent_idx, scores, result_dicts)
        
        if only_highest_beam:
                    
            cleaned_weight_matrix = cleaned_weight_matrix[:,0,:,:,:,:][:,None,:,:,:,:]
            cleaned_score_matrix = cleaned_score_matrix[:,0,:][:,None,:]
            
            result_dicts["scores_array"] = result_dicts["scores_array"][:,0,:][:,None,:]
            result_dicts["token_beam_array"] = result_dicts["token_beam_array"][:,0,:][:,None,:]
            result_dicts["beam_length"] = result_dicts["beam_length"][:,0][:,None]
            
        
        shape = cleaned_weight_matrix.shape
    
        tmp_scores_array = np.zeros(shape=[shape[0],shape[1], max_beam_length])
        tmp_scores_array[:,:,:weights.shape[2]] = result_dicts["scores_array"]
        
        tmp_token_array = np.zeros(shape=[shape[0],shape[1], max_beam_length])
        tmp_token_array[:,:,:weights.shape[2]] = result_dicts["token_beam_array"]
        
        result_dicts["scores_array"] = tmp_scores_array
        result_dicts["token_beam_array"] = tmp_token_array
        
        
        tmp_weights = np.zeros(shape=[shape[0],shape[1],max_beam_length,shape[3],shape[4],shape[5]])
        tmp_scores = np.zeros(shape=[shape[0],shape[1],max_beam_length])
        
        tmp_weights[:,:,:weights.shape[2],:,:,:] = cleaned_weight_matrix
        tmp_scores[:,:,:weights.shape[2]] = cleaned_score_matrix
        
        weights_list.append(tmp_weights)
        result_dicts_list.append(result_dicts)
        scores_list.append(tmp_scores)
        
    cleaned_weight_matrix = np.concatenate(weights_list)
    cleaned_score_matrix = np.concatenate(scores_list)
    
    result_dict = dict()
    for key in ['longest_beam_array', 'data_ids', 'summary_beam_list', 'number_of_textual_units', 'scores_array', 'token_beam_array', 'beam_length']:

        result_dict[key] = np.concatenate([result_dicts_list[i][key] for i in range(len(batch_directories))])
        
    return cleaned_weight_matrix, cleaned_score_matrix, result_dict
    
    

if __name__ == '__main__':
    main()