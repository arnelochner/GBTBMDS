import numpy as np
from tqdm import tqdm


def custom_function(y, fun, sentences_ending_information, max_num_sentences, shapes):
    i, x = y
    ex_num = int(i / shapes[4])
    beam_num = i % shapes[4]
    sentence_information = sentences_ending_information[ex_num, beam_num].astype(
        "int")
    beam = x.reshape(shapes[3], shapes[0], shapes[1], shapes[2])

    result_matrix = np.zeros(
        shape=[max_num_sentences-1, shapes[0], shapes[1], shapes[2]])

    for i in range(0, len(sentence_information)-1):
        sentence_start = sentence_information[i]
        sentence_end = sentence_information[i+1]
        tmp = beam[sentence_start:sentence_end, :, :, :]
        result_matrix[i, :, :, :] = np.apply_along_axis(
            fun, 0, tmp).reshape(1, shapes[0], shapes[1], shapes[2])

    return result_matrix


def aggregate_weight_information_for_sentences(cleaned_weight_matrix, result_dict, eoq_token=8):
    
    num_examples, num_beams, num_steps, num_decoding_layer, num_multi_head, num_paragraphs = cleaned_weight_matrix.shape

    sentences_ending_information = np.array([np.append(np.insert(np.where(beam == eoq_token), 0, 0, axis=1).reshape(-1,), [
                                            result_dict["beam_length"][i, j]], axis=0) for i, ex in enumerate(result_dict["token_beam_array"].astype("int")) for j, beam in enumerate(ex)]).reshape(num_examples, num_beams)

    max_num_sentences = np.max(
        [len(beam) for ex in sentences_ending_information for beam in ex])

    reshaped_matrix = cleaned_weight_matrix.reshape(num_examples*num_beams, -1)

    functions = [{"name": "Mean", "fun": np.mean}]#,
                 #{"name": "Median", "fun": np.median}]

    res_dict = dict()

    for fun in tqdm(functions):
        r = list(map(lambda x: custom_function(
            x, fun["fun"], sentences_ending_information, max_num_sentences, (num_decoding_layer, num_multi_head, num_paragraphs, num_steps, num_beams)), tqdm(enumerate(reshaped_matrix))))
        res_dict[fun["name"]] = np.array(r).reshape(num_examples, num_beams, max_num_sentences-1,
                                                    num_decoding_layer, num_multi_head, num_paragraphs)

    return res_dict


"""
Backup function
def aggregate_weight_information_for_sentences(cleaned_weight_matrix,sentences_ending_information):
      
    functions=[{"name": "Mean", "fun": np.mean}, {"name": "Median", "fun": np.median}]
    res_dict = dict()
    for fun in functions:
        res_list_fun = []
        for i,ex in enumerate(cleaned_weight_matrix):
            res_list = []
            for j,beam in enumerate(ex):
                result_list = my_aggregate_function(beam,sentences_ending_information[i,j].astype("int"), fun["fun"])
                res_list.append(result_list)
            res_list_fun.append(np.array(res_list))
        res_dict[fun["name"]] = res_list_fun
    return res_dict
    
def my_aggregate_function(beam,sentence_information, fun):
    result_list = []
    for i in range(len(sentence_information)-1):
        sentence_start = sentence_information[i]
        sentence_end = sentence_information[i+1]
        tmp = beam[sentence_start:sentence_end, :, :,:]
        result = np.apply_along_axis(fun, 0, tmp)
        result_list.append(result)
    return np.array(result_list)

"""
