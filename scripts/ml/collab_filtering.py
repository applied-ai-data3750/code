def dict_per_user(interset_word_list: list, interests: list) -> dict:
    """
    function that takes a list of interests words, and returns a dict 
    with the interests as keys and the number of times the interest 
    appears in the list as values


    Args:
        `interset_word_list` (list): list of words per user

    Returns:
        `tempdict`: dict of commits per interest in float values
    """
    
    tempdict = dict.fromkeys(interests, 0)

    for ele in interset_word_list:
        tempdict[ele] += 1

    return tempdict