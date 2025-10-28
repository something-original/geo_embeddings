def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

def flatten(a):
    return [item for sublist in a for item in sublist]