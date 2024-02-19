def flatten(lst: list) -> list:
    """
    Flattens a nested list that contains sub-lists.

    Args:
        :param lst: A nested list.
        :type lst: list

    Returns:
        :return: An unnested list.
        :rtype: list

    Example:
        > nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        > flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [item for sublist in lst for item in sublist]