import xxhash


def xxhash_encoder(value,seed=10):
    value=str(value)
    return xxhash.xxh64(value,seed=seed).intdigest()



if __name__ == '__main__':
    print(xxhash_encoder(123))