import numpy as np
import pickle


def reconstruct_attention_weights(attention_weights, parent_idx, scores, result_dict):
    """
    Function used to reconstruct output from graphsum model. The issue with the output from graphsum is, that beams/examples
    can be finished before the maximum token_length is reached. This causes the shape of the output to not be a constant value
    therefore padding is required to force each output to have the same shape. 
    By padding the outputs you cannot assume that the first example in this output corresponds to the actual first example.
    This function is able to reconstruct this mapping with the meta infromation stored in the result_dict.

    Args:
    - attention_weights(np.array: shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Padded Matrix which should be "reconstructed"

    - parent_idx(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Padded Parent_Idx matrix which should be "reconstructed"

    - scores(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Padded Scores matrix which should be "reconstructed"

    - result_dict(dict(str,obj)):
        Dictonary with the following keys:
            'scores_array': scores_array which are the output of the beam_search_decoding layer in paddlepaddle
            'longest_beam_array': Array which stores information how long the longest beam is for each example
            'summary_beam_list': List containing all generated summaries for the example in string form.
            'token_beam_array': Array containing all generated summaries for the example in token form.

    Returns:
    - attention_weights_matrix(np.array: shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Reconstructed Attention Weight array

    - scores_matrix(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Reconstructed Scores array

    - parent_idx_matrix(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Reconstructed parent_idx array

    """

    # Retrieve shape information
    examples, beam_size, number_steps, _, _, _ = attention_weights.shape

    # Retrieve arrays/lists from dict
    finished_scores_array = result_dict["scores_array"]
    longest_beam_array = (result_dict["longest_beam_array"] - 1).astype("int")
    summary_beam_list = np.array(result_dict["summary_beam_list"]).reshape(
        attention_weights.shape[:2])
    token_beam_array = result_dict["token_beam_array"]

    # Create arrays where reconstructed arrays are stored
    attention_weights_matrix = np.zeros(shape=attention_weights.shape)
    parent_idx_matrix = -np.ones(shape=parent_idx.shape)
    scores_matrix = -np.ones(shape=scores.shape)

    # Step-Number until all examples are still "valid"
    first_finished = np.min(longest_beam_array)

    # Simply copy values until this step-number, because the mapping is still valid
    attention_weights_matrix[:, :, :first_finished, :,
                             :] = attention_weights[:, :, :first_finished, :, :]
    parent_idx_matrix[:, :,
                      :first_finished] = parent_idx[:, :, :first_finished]
    scores_matrix[:, :, :first_finished] = scores[:, :, :first_finished]

    # Iterate over all output steps
    for i in range(first_finished, number_steps):

        # For each iteration check which examples are still "valid"
        remaining_examples = np.delete(
            np.arange(0, examples, 1), np.where(longest_beam_array < i))

        # Work with cum_sum in order to calculate shift, of examples that are still "valid"
        # E.g if examples 0 and 1 are already finished example 2 is the first example in attention_weights for this interation
        tmp_cumsum = np.delete(
            np.cumsum(1*(longest_beam_array < i)), np.where(longest_beam_array < i))

        # Copy corresponding values from output arrays and use 'tmp_cumsum' as shifting values to recreate mapping
        scores_matrix[remaining_examples, :,
                      i] = scores[remaining_examples-tmp_cumsum, :, i]
        parent_idx_matrix[remaining_examples, :,
                          i] = parent_idx[remaining_examples-tmp_cumsum, :, i]
        attention_weights_matrix[remaining_examples, :, i, :,
                                 :] = attention_weights[remaining_examples-tmp_cumsum, :, i, :, :]

    return attention_weights_matrix, scores_matrix, parent_idx_matrix


def transform_parent_idx(parent_idx_vector):
    """
    Helper Function to transform parent_idx_vector's values to be in range of [0,4] instead of row_number*beam_size + [0,4]
    Function is applied with np.apply_along_axis.

    Args:

    - parent_idx_vector (np.array: shape=[beam_size])

    Returns:

    - Transformed parent_idx_vector with values in range of [0,4]
    """
    return parent_idx_vector - (int(parent_idx_vector[0] / len(parent_idx_vector)) * len(parent_idx_vector))


def transform_attention_weights_decoder(attention_weights, score_matrix, parent_idx_matrix, end_array):
    """
    Function used to replicate beam_search_decoder in order to transform attention_weights and score_matrix so 
    decoded_weight_matrix[0,0,:,:,:,:] are all values corresponding to the first beam of the first example.

    Args:

    - attention_weights (np.array, shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Array which contains attention_weights values which should be transformed

    - score_matrix (np.array, shape=[
        num_examples,beam_size,max_out_len]
        ): Array which contains scores values which should be transformed

    - parent_idx (np.array, shape=[
        num_examples,beam_size,max_out_len]
        ): Array which contains information, to recreate the beam paths

    - end_array (list, len=num_examples): Contains information, how long the longest beam of each example is

    Returns:

    - decoded_weight_matrix (np.array, shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Transformed Weight Matrix, where the beams are "recreated"

    - decoded_score_matrix (np.array, shape=[
        num_examples,beam_size,max_out_len]
        ): Transformed Score Matrix, where the beams are "recreated"

    """

    # Extract Shapes for Numpy Arrays
    examples, beam_size, number_steps, _, _, _ = attention_weights.shape

    # Decoder Matrix where new values are stored
    decoded_weight_matrix = np.zeros(shape=attention_weights.shape)

    # Decoder Matrix where new values are stored
    decoded_score_matrix = np.zeros(shape=score_matrix.shape)

    # Check for first iteration which examples have beams with beam_length == max_output_len
    remaining_examples = np.delete(
        np.arange(0, examples, 1), np.where((end_array < number_steps)))

    # Create Matrix, which contains Parent_idx for all beams of all examples
    # If all beams are finished for a beam [0,1,2,3,4] is used to retrieve the parent_ids from parent_idx matrix
    current_transformed_parent_idx_matrix = np.tile(
        np.arange(0, beam_size), examples).reshape(-1, beam_size)

    # Looping over all steps with inverse order
    for i in range(number_steps):
        # Current Step number to loop in inverse order
        current_step = number_steps - i - 1

        # Retrieve examples, which have a beam with beam_length > current_step
        # shape = [# of remaining_examples]
        remaining_examples = np.delete(
            np.arange(0, examples, 1), np.where((end_array < current_step)))

        # Retrieve corresponding parent_idx from last loop for remaining examples
        # shape = [# of remaining_examples, beam_size]
        current_transformed_parent_idx = current_transformed_parent_idx_matrix[
            remaining_examples, :]

        # Retrieve Weight Values from Attention-Weight Matrix based on remaining examples and their corresponding parent_ids
        # shape = [# of remaining_examples * beam_size, num_decoding_layer, num_multi_heads, num_paragraphs]
        weight_values = attention_weights[np.repeat(
            remaining_examples, beam_size), current_transformed_parent_idx.reshape(-1,), current_step]

        # Retrieve Score Values from Score-Weight Matrix based on remaining examples and their corresponding parent_ids
        # shape = [# of remaining_examples * beam_size]
        score_values = score_matrix[np.repeat(
            remaining_examples, beam_size), current_transformed_parent_idx.reshape(-1,), current_step]

        # Update Values in Decoder Matrix
        decoded_weight_matrix[np.repeat(remaining_examples, beam_size), np.tile(
            np.arange(0, beam_size), len(remaining_examples)), current_step] = weight_values

        # Update Values in Decoder Matrix
        decoded_score_matrix[np.repeat(remaining_examples, beam_size), np.tile(
            np.arange(0, beam_size), len(remaining_examples)), current_step] = score_values

        # Calculate Parent-Ids for next iteration based on current values for parent_idx
        # shape = [# of remaining_examples * beam_size]
        current_parent_idx = parent_idx_matrix[np.repeat(
            remaining_examples, beam_size), current_transformed_parent_idx.reshape(-1,).astype("int"), current_step]

        # Transform current_parent_idx, where each row contains value from [0-4] + row_number*beam_size to range [0-4]
        # shape = [# of remaining_examples * beam_size]
        current_transformed_parent_idx = np.apply_along_axis(
            transform_parent_idx, 1, current_parent_idx.reshape(-1, beam_size)).astype("int")

        # Update values in parent_idx matrix, which is used for next iteration
        current_transformed_parent_idx_matrix[remaining_examples,
                                              :] = current_transformed_parent_idx

    return decoded_weight_matrix, decoded_score_matrix


def transform_attention_weights(attention_weights, parent_idx, scores, result_dict):
    """
    Wrapper Function, which firstly reconstructs mapping of examples in arrays and afterwards transforms
    these arrays based on decoding algorithm to recreate beams of examples.

    Args:
    - attention_weights(np.array: shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Padded Matrix which should be "reconstructed"

    - parent_idx(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Padded Parent_Idx matrix which should be "reconstructed"

    - scores(np.array: shape=[
        num_examples,beam_size,max_out_len]
        ): Padded Scores matrix which should be "reconstructed"

    - result_dict(dict(str,obj)):
        Dictonary with the following keys:
            'scores_array': scores_array which are the output of the beam_search_decoding layer in paddlepaddle
            'longest_beam_array': Array which stores information how long the longest beam is for each example
            'summary_beam_list': List containing all generated summaries for the example in string form.
            'token_beam_array': Array containing all generated summaries for the example in token form.

    Returns:

    - decoded_weight_matrix (np.array, shape=[
        num_examples,beam_size,max_out_len,num_decoding_layer,num_multi_head,num_paragraphs]
        ): Transformed Weight Matrix, where the beams are "recreated"

    - decoded_score_matrix (np.array, shape=[
        num_examples,beam_size,max_out_len]
        ): Transformed Score Matrix, where the beams are "recreated"

    """

    attention_weights_matrix, scores_matrix, parent_idx_matrix = reconstruct_attention_weights(
        attention_weights, parent_idx, scores, result_dict)

    decoded_weight_matrix, decoded_score_matrix = transform_attention_weights_decoder(
        attention_weights_matrix, scores_matrix, parent_idx_matrix, result_dict["longest_beam_array"]-1)

    sorted_weight_matrix, sorted_score_matrix = sort_matrixes(
        decoded_weight_matrix, decoded_score_matrix, result_dict["longest_beam_array"]-1)

    cleaned_weight_matrix, cleaned_score_matrix = cleanup_matrix(sorted_weight_matrix, sorted_score_matrix,
                                                                 result_dict["beam_length"])

    return cleaned_weight_matrix, cleaned_score_matrix


def sort_matrixes(decoded_weight_matrix, decoded_score_matrix, longest_beam_array):

    num_examples, num_beams, _ = decoded_score_matrix.shape

    sort_information = np.argsort(
        -decoded_score_matrix[range(0, num_examples), :, longest_beam_array.astype("int")])

    sorted_score_matrix = decoded_score_matrix[np.repeat(np.array(list(
        range(0, num_examples)))[:, np.newaxis], num_beams, axis=1), sort_information, :]

    sorted_weight_matrix = decoded_weight_matrix[np.repeat(np.array(list(range(
        0, num_examples)))[:, np.newaxis], num_beams, axis=1), sort_information, :, :, :, :]

    return sorted_weight_matrix, sorted_score_matrix


def cleanup_matrix(sorted_weight_matrix, sorted_score_matrix, beam_length):

    num_examples, num_beams, num_steps = sorted_score_matrix.shape

    def create_index_information(x):
        return [list(range(int(i), num_steps)) for i in x]

    example_index = np.repeat(list(range(0, num_examples)), np.sum(
        num_steps - beam_length, axis=1).astype("int"))

    indexs = list(map(create_index_information, beam_length))

    value_indexs = [
        index for elem in indexs for beam in elem for index in beam]

    beam_index = np.repeat(np.tile(list(range(0, num_beams)), num_examples),
                           (num_steps - beam_length).astype("int").reshape(-1,))

    cleaned_score_matrix = sorted_score_matrix.copy()

    cleaned_weight_matrix = sorted_weight_matrix.copy()

    cleaned_score_matrix[example_index, beam_index, value_indexs] = 0

    cleaned_weight_matrix[example_index, beam_index, value_indexs, :, :, :] = 0

    return cleaned_weight_matrix, cleaned_score_matrix
