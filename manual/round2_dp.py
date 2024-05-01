lookup = {
    'pizza': {'pizza': 1, 'wasabi': 0.48, 'snowball': 1.52, 'shells': 0.71},
    'wasabi': {'pizza': 2.05, 'wasabi': 1, 'snowball': 3.26, 'shells': 1.56},
    'snowball': {'pizza': 0.64, 'wasabi': 0.3, 'snowball': 1, 'shells': 0.46},
    'shells': {'pizza': 1.41, 'wasabi': 0.61, 'snowball': 2.08, 'shells': 1}
}

oid = {
    0: 'pizza',
    1: 'wasabi',
    2: 'snowball',
    3: 'shells'
}

dp = [[(-1.0, []) for _ in range(4)] for _ in range(6)]

def recur(depth, item):
    if dp[depth][item][0] != -1:
        return dp[depth][item]
    else:
        if depth == 0:
            dp[depth][item] = (2000 * lookup['shells'][oid[item]], ['shells', oid[item]])
            return dp[depth][item]
        else:
            mv = 0.0
            lst = []
            ind = -1
            for i in range(4):
                tv, tlst = recur(depth - 1, i)
                cv = tv * lookup[oid[i]][oid[item]]
                if cv > mv:
                    mv = cv
                    lst = tlst + [oid[item]]
            dp[depth][item] = (mv, lst)
            return dp[depth][item]

lst = []
max_val = 0

max_val, lst = recur(4, 3)


print(lst, max_val)
