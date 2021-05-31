def most_frequent(ids):
    counter = 0
    most_id = ids[0]
    for id_ in ids:
        curr_frequency = ids.count(id_)
        if curr_frequency > counter:
            counter = curr_frequency
            most_id = id_
    indices = [i for i, x in enumerate(ids) if x == most_id]
    
    return int(most_id), counter, indices


def most_frequent_with_scores(ids, scores):
    counter = 0
    most_id = ids[0]
    for id_ in ids:
        curr_frequency = ids.count(id_)
        if curr_frequency > counter:
            counter = curr_frequency
            most_id = id_
        elif curr_frequency == counter and curr_frequency > 1:
            indices_1 = [i for i, x in enumerate(ids) if x == most_id]
            total_scores_1 = sum([scores[index] for index in indices_1])
            indices_2 = [i for i, x in enumerate(ids) if x == id_]
            total_scores_2 = sum([scores[index] for index in indices_2])
            if total_scores_2 < total_scores_1:
                most_id = id_

    indices = [i for i, x in enumerate(ids) if x == most_id]

    return int(most_id), counter, indices