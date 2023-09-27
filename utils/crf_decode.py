import torch
from typing import Optional

def decode(emissions: torch.Tensor,
        # num_tags,
        transitions,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
        batch_first = True,
        mask = None):
    """Find the most likely tag sequence using Viterbi algorithm.

    Args:
        emissions (`~torch.Tensor`): Emission score tensor of size
            ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length, num_tags)`` otherwise.
        mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
            if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

    Returns:
        List of list containing the best tag sequence for each batch.
    """
    if not isinstance(emissions, torch.Tensor):
        emissions = torch.Tensor(emissions)
    if not isinstance(transitions, torch.Tensor):
        transitions = torch.Tensor(transitions)
    if not isinstance(start_transitions, torch.Tensor):
        start_transitions = torch.Tensor(start_transitions)
    if not isinstance(end_transitions, torch.Tensor):
        end_transitions = torch.Tensor(end_transitions)
    num_tags = emissions.shape[-1]
    _validate(emissions, num_tags, mask=mask)
    if mask is None:
        mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

    if batch_first:
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

    return _viterbi_decode(emissions, num_tags, mask, transitions, start_transitions, end_transitions)

def _validate(
        emissions: torch.Tensor,
        num_tags,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
        batch_first = True,):
    if emissions.dim() != 3:
        raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
    if emissions.size(2) != num_tags:
        raise ValueError(
            f'expected last dimension of emissions is {num_tags}, '
            f'got {emissions.size(2)}')

    if tags is not None:
        if emissions.shape[:2] != tags.shape:
            raise ValueError(
                'the first two dimensions of emissions and tags must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

    if mask is not None:
        if emissions.shape[:2] != mask.shape:
            raise ValueError(
                'the first two dimensions of emissions and mask must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
        no_empty_seq = not batch_first and mask[0].all()
        no_empty_seq_bf = batch_first and mask[:, 0].all()
        if not no_empty_seq and not no_empty_seq_bf:
            raise ValueError('mask of the first timestep must all be on')

def _viterbi_decode(emissions: torch.FloatTensor,
                    num_tags,
                    mask: torch.ByteTensor,
                    transitions,
                    start_transitions: torch.Tensor,
                    end_transitions: torch.Tensor,
                    ):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert emissions.size(2) == num_tags
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list