import hashlib


def set_seed(string, seed_func):
    """
    This function hashes `string` to an integer and set as seed using `seed_func`.

    Args:
        string (str): the string to be hashed and used as seed.
        seed_func (Callable): "random.seed" or "torch.manual_seed".
    Returns:
        seed (int): the hash result of given string.
    """
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    if seed_func is not None:
        seed_func(seed)
    return seed
