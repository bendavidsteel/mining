import numpy as np
import torch

def weighted_exponential_smoothing(opinion_sequences, mask_sequences, halflife):
    opinions = np.zeros((len(opinion_sequences), opinion_sequences[0].shape[1]))
    for user_idx in range(len(opinion_sequences)):
        opinion_sequence = opinion_sequences[user_idx]
        mask_sequence = mask_sequences[user_idx].reshape(-1,1)
        num_points = opinion_sequence.shape[0]
        alpha = 1 - np.exp(-np.log(2) / halflife)
        attention_vector = np.array([(1 - alpha) ** i for i in range(num_points)]).reshape(-1,1)
        opinion = np.sum(opinion_sequence * attention_vector * mask_sequence, axis=0) / np.sum(attention_vector * mask_sequence, axis=0)
        opinions[user_idx] = opinion
    return opinions

def degroot(user_state, new_content, attention):
    return user_state + torch.bmm(attention.unsqueeze(1), new_content).squeeze(1)

def friedkin_johnsen(original_user_state, content, stubbornness):
    return ((1 - stubbornness) * original_user_state) + (stubbornness * torch.sum(content))

def bounded_confidence(user_state, new_content, confidence_interval):
    diff = new_content - user_state
    # TODO replace with logistic?
    mask = torch.norm(torch.abs(diff), dim=1) < confidence_interval
    return user_state + ((1 / torch.sum(mask)) * torch.sum(mask.unsqueeze(1) * diff, axis=0))

def stochastic_bounded_confidence(user_state, new_content, content_mask, exponent):
    # compares prob of interaction between a number of people
    unsq_content_mask = content_mask.unsqueeze(2)
    # we can't use cdist because we can't mask in cdist, and unmasked cdist would be computationally inefficient
    content_dist = torch.cdist(user_state.unsqueeze(1), new_content * unsq_content_mask).squeeze(1) * content_mask

    # TODO use MaskedTensor
    numerator = torch.pow(content_dist + ((1 - content_mask) * 1), -1 * exponent.unsqueeze(1)) * content_mask
    numerator = torch.clamp_max(numerator, 100)
    return numerator

def stochastic_bounded_confidence_probs(user_state, new_content, content_mask, exponent):
    numerator = stochastic_bounded_confidence(user_state, new_content, content_mask, exponent)
    denominator = torch.sum(numerator, dim=1, keepdims=True)

    batch_mask = content_mask.any(dim=1)
    zero_mask = (denominator != 0).squeeze(1) # to avoid division by zero
    mask = batch_mask * zero_mask

    probs = torch.zeros(numerator.shape, dtype=torch.float32)
    probs[mask] = numerator[mask] / denominator[mask]

    return probs

def stochastic_bounded_confidence_probs_plus_null(user_state, new_content, content_mask, exponent):
    raise NotImplementedError()

def stochastic_bounded_confidence_bernoulli_average(user_state, new_content, exponent, n, avg_content):
    # compares prob of interaction between a number of people
    numerator = torch.pow(torch.abs(user_state - new_content), -1 * exponent)
    return numerator / (numerator + ((n-1) * torch.pow(torch.abs(user_state - avg_content), -1 * exponent)))

def sbc_choice(user_state, new_content, content_mask, exponent):
    probs = stochastic_bounded_confidence_probs(user_state, new_content, content_mask, exponent)
    chosen_content = torch.distributions.Categorical(probs=probs).sample()
    content_diff = new_content[torch.arange(len(new_content)), chosen_content] - user_state

    # apply mask functionality
    # if none of the content is valid, we return a zero content diff and a null content index (using -1)
    batch_mask = content_mask.any(dim=1)
    content_diff *= batch_mask.unsqueeze(1)
    chosen_content[batch_mask.logical_not()] = -1 # -1 is the null content

    return content_diff, chosen_content

def sbc_choice_plus_null(user_state, new_content, content_mask, exponent):
    probs = stochastic_bounded_confidence_probs_plus_null(user_state, new_content, content_mask, exponent)
    chosen_content = torch.distributions.Categorical(probs=probs).sample()
    return new_content[torch.arange(len(new_content)), chosen_content] - user_state, chosen_content
