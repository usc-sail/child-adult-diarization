def get_timestamps(labels):
    """
    Input: list of labels at each 20ms frame
    Output: list of tuples of start and end time of each label
    """
    child, adult, overlap = [], [], []
    idx, time = 0, 0
    while idx < len(labels):
        if labels[idx] == 1:
            start = round(time, 3)
            while idx < len(labels) and labels[idx] == 1:
                idx += 1
                time += 0.02
            end = round(time, 3)
            child.append((start, end))
        elif labels[idx] == 2:
            start = round(time, 3)
            while idx < len(labels) and labels[idx] == 2:
                idx += 1
                time += 0.02
            end = round(time, 3)
            adult.append((start, end))
            
        elif labels[idx] == 3:
            start = round(time, 3)
            while idx < len(labels) and labels[idx] == 3:
                idx += 1
                time += 0.02
            end = round(time, 3)
            overlap.append((start, end))
        else:
            idx += 1
            time += 0.02
    return child, adult, overlap