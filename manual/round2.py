hashmap = {
    'pizza': {'pizza': 1, 'wasabi': 0.48, 'snowball': 1.52, 'shells': 0.71},
    'wasabi': {'pizza': 2.05, 'wasabi': 1, 'snowball': 3.26, 'shells': 1.56},
    'snowball': {'pizza': 0.64, 'wasabi': 0.3, 'snowball': 1, 'shells': 0.46},
    'shells': {'pizza': 1.41, 'wasabi': 0.61, 'snowball': 2.08, 'shells': 1}
}

def recurse(prev, currency, depth):
    if depth == 5 and currency == 'shells':
        return hashmap[prev][currency], [currency]
    elif depth == 5:
        return 0, []

    if prev is None:
        max_value, max_path = max((recurse(currency, c, depth + 1) for c in hashmap['shells'].keys()), key=lambda x: x[0])
        return max_value, [currency] + max_path
    else:
        max_value, max_path = max((recurse(currency, c, depth + 1) for c in hashmap[currency].keys()), key=lambda x: x[0])
        return hashmap[prev][currency] * max_value, [currency] + max_path

max_value, max_path = recurse(None, 'shells', 0)
print("Maximum value:", max_value)
print("Path:", max_path)
