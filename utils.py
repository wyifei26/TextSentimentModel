def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'🌟{func.__name__}: {(time.time() - start):.2f}s🌟')
        return result

    return wrapper
