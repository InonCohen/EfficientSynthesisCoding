import random
import numpy as np


# for the sake of code clarity, there is not a real need in python to use a swap function
def swap(a, b): 
    return b, a


def early_lcs_with_indexes(s1, s2):     #returns the lcs with the lowest indexes, and its indexes
    m, n = len(s1), len(s2)
    dp = [[('', (-1, -1)) for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0] + s1[i - 1], (i - 1, j - 1))
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=lambda x: len(x[0]))
    # extract the LCS and indexes
    lcs, (i, j) = dp[m][n]
    idx1 = []
    idx2 = []
    i, j = 0, 0
    for k in range(len(lcs)):
        while i < len(s1) and s1[i] != lcs[k]:
            i += 1
        while j < len(s2) and s2[j] != lcs[k]:
            j += 1
        idx1.append(i)
        idx2.append(j)
        i += 1
        j += 1
    return lcs, idx1, idx2


def late_lcs(str1, str2):           #returns the lcs with the highest indexes, and its indexes
    m = len(str1)
    n = len(str2)
    # Initialize an (m+1)x(n+1) matrix of zeros
    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
    # Compute the LCS matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i][j - 1], lcs_matrix[i - 1][j])
    # Traverse the matrix to construct the LCS string
    lcs_string = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs_string = str1[i - 1] + lcs_string
            i -= 1
            j -= 1
        elif lcs_matrix[i - 1][j] > lcs_matrix[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs_string


def indexes_of_lcs(str1, str2):   #returns indexes of lcs, probably greatest indexes // todo: what on earth is that?
    m = len(str1)
    n = len(str2)
    # Initialize an (m+1)x(n+1) matrix of zeros
    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
    # Compute the LCS matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i][j - 1], lcs_matrix[i - 1][j])
    # Traverse the matrix to construct the LCS string and its indexes
    lcs_string = ""
    s1_indexes = []
    s2_indexes = []
    i = m
    j = n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs_string = str1[i - 1] + lcs_string
            s1_indexes.insert(0, i - 1)
            s2_indexes.insert(0, j - 1)
            i -= 1
            j -= 1
        elif lcs_matrix[i - 1][j] > lcs_matrix[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return s1_indexes, s2_indexes


def leading_indexes1(str1, str2, str3):   # maps strings to lcs indexes# (greatest indexes) //todo: again, what does that mean?
    # Compute indexes of LCS of every pair
    lcs_dict = {'1in12': (indexes_of_lcs(str1, str2))[0], '2in12': (indexes_of_lcs(str1, str2))[1],
                '1in13': (indexes_of_lcs(str1, str3))[0], '3in13': (indexes_of_lcs(str1, str3))[1],
                '2in23': (indexes_of_lcs(str2, str3))[0], '3in23': (indexes_of_lcs(str2, str3))[1]}
    # create a dictionary to store the number of indexes for each string's LCS
    indexes_count = {
        '1in12': len(lcs_dict['1in12']),
        '2in12': len(lcs_dict['2in12']),
        '1in13': len(lcs_dict['1in13']),
        '3in13': len(lcs_dict['3in13']),
        '2in23': len(lcs_dict['2in23']),
        '3in23': len(lcs_dict['3in23'])
    }
    # sort the dictionary by the number of indexes
    sorted_indexes = sorted(indexes_count.items(), key=lambda x: x[1], reverse=True)
    # create a new dictionary with only the top two strings and their indexes
    if (sorted_indexes[0][0] == '1in12' and sorted_indexes[1][0] == '2in12') or \
            (sorted_indexes[0][0] == '1in13' and sorted_indexes[1][0] == '3in13') or \
            (sorted_indexes[0][0] == '2in23' and sorted_indexes[1][0] == '3in23'):
        top_two_strings = {
            sorted_indexes[0][0]: lcs_dict[sorted_indexes[0][0]],
            sorted_indexes[1][0]: lcs_dict[sorted_indexes[1][0]]
        }
    else:
        top_two_strings = {}
    return top_two_strings


def leading_indexes2(str1, str2, str3):  # maps strings to lcs indexes (smallest indexes)
    lcs_dict = {}
    # Compute indexes of LCS of str1 and str2
    lcs12, lcs_dict['1in12'], lcs_dict['2in12'] = early_lcs_with_indexes(str1, str2)
    # Compute indexes of LCS of str1 and str3
    lcs13, lcs_dict['1in13'], lcs_dict['3in13'] = early_lcs_with_indexes(str1, str3)
    # Compute indexes of LCS of str2 and str3
    lcs23, lcs_dict['2in23'], lcs_dict['3in23'] = early_lcs_with_indexes(str2, str3)
    # create a dictionary to store the number of indexes for each string's LCS
    indexes_count = {
        '1in12': len(lcs_dict['1in12']),
        '2in12': len(lcs_dict['2in12']),
        '1in13': len(lcs_dict['1in13']),
        '3in13': len(lcs_dict['3in13']),
        '2in23': len(lcs_dict['2in23']),
        '3in23': len(lcs_dict['3in23'])
    }
    # sort the dictionary by the number of indexes
    sorted_indexes = sorted(indexes_count.items(), key=lambda x: x[1], reverse=True)
    # create a new dictionary with only the top two strings and their indexes
    if (sorted_indexes[0][0] == '1in12' and sorted_indexes[1][0] == '2in12') or \
            (sorted_indexes[0][0] == '1in13' and sorted_indexes[1][0] == '3in13') or \
            (sorted_indexes[0][0] == '2in23' and sorted_indexes[1][0] == '3in23'):
        top_two_strings = {
            sorted_indexes[0][0]: lcs_dict[sorted_indexes[0][0]],
            sorted_indexes[1][0]: lcs_dict[sorted_indexes[1][0]]
        }
    else:
        top_two_strings = {}
    return top_two_strings


def count_different_letters(s1, s2, s3):
    n = max(len(s) for s in (s1, s2, s3))
    result = []
    for i in range(n):
        letters = set()
        for s in (s1, s2, s3):
            if i < len(s):
                letters.add(s[i])
        result.append(len(letters))
    return result


def separate_strings(s1, s2, s3):
    n = len(s1)
    a = []
    # First n pairs: (s1[i], s2[i])
    for i in range(n):
        a.append(set((s1[i], s2[i])))
        while len(a[i]) == 1:
            a[i].add(random.choice(['A', 'C', 'G', 'T']))
    # Last n pairs: (s3[n-i], c) where c is different than s3[i]
    chars = ['A', 'C', 'G', 'T']
    for i in range(n):
        c = chars[0]
        if s3[i] != c:
            a.append((s3[n - i - 1], c))
        else:
            a.append((s3[n - i - 1], chars[1]))
    return a

def create_cycle_by_first_index(s1, s2, s3, c1, c2, c3, cycles, current_cycle, first_index):
    if s2[c2] == s1[c1] or s3[c3] == s1[c1]:
        cycles.append(set(s1[c1]) | set(s2[c2]) | set(s3[c3]))
        c1 += 1
        c2 += 1
        c3 += 1
        current_cycle += 1
        return
    else:
        second_index = min(c2, c3)
        if second_index == c2:
            cycles.append(set(s1[c1]) | set(s2[c2]))
            c1 += 1
            c2 += 1
            current_cycle += 1
            return
        else:
            cycles.append(set(s1[c1]) | set(s3[c3]))
            c1 += 1
            c3 += 1
            current_cycle += 1
            return


def create_sets(string1, string2, string3, instance):
    x = 0
    if instance == 1:
        x = leading_indexes1(s1, s2, s3)
    if instance == 2:
        x = leading_indexes2(s1, s2, s3)
    if x == {}:
        return separate_strings(s1, s2, s3)
    # print(x)
    lcs_strings = [key[0] for key in x.keys()]
    lcs_indexes = [value for value in x.values()]
    lcs1_indexes = lcs_indexes[0]
    lcs2_indexes = lcs_indexes[1]
    a = int(lcs_strings[0])
    b = int(lcs_strings[1])
    if a != 1:  # s23
        string1, string2 = swap(string1, string2)
        string2, string3 = swap(string2, string3)
    elif b == 3:
        string2, string3 = swap(string2, string3)
    # the lcs is of s1 and s2, and the indexes of the lcs in s1 is lcs1_indexes, and the indexes of the lcs in s2 is lcs2_indexes
    # print(lcs1_indexes, lcs2_indexes)
    # print(lcs_strings)
    # print(lcs_indexes)
    n = len(s1)  # assume all strings are of length n
    c1, c2, c3, current_cycle = 0, 0, 0, 0  # c1 iterates over s1, c2 over s2, c3 over s3
    cycles = []
    while c1 < n and c2 < n and c3 < n:
        letters_of_cycle = set(s1[c1]) | set(s2[c2]) | set(s3[c3])
        diff_letters_in_cycle = len(letters_of_cycle)
        if diff_letters_in_cycle <= 2:
            cycles.append(letters_of_cycle)
            c1 += 1
            c2 += 1
            c3 += 1
            current_cycle += 1
            continue
        # in the current cycle there are three different letters
        else:
            # check if there are letters from the lcs to be synthesized in the current cycle
            # if they are singular - synthesize the other two letters and continue to the next cycle
            if c1 in lcs1_indexes and c2 not in lcs2_indexes:
                cycles.append(set(s2[c2]) | set(s3[c3]))
                c2 += 1
                c3 += 1
                current_cycle += 1
                continue
            if c1 not in lcs1_indexes and c2 in lcs2_indexes:
                cycles.append(set(s1[c1]) | set(s3[c3]))
                c1 += 1
                c3 += 1
                current_cycle += 1
                continue
        # in the current cycle there are three different letters, and if there are lcs letters, they're a pair
        first_index = min(c1, c2, c3)  # first index to be synthesized
        if first_index == c1:
            if s2[c2] == s1[c1] or s3[c3] == s1[c1]:
                cycles.append(set(s1[c1]) | set(s2[c2]) | set(s3[c3]))
                c1 += 1
                c2 += 1
                c3 += 1
                current_cycle += 1
                continue
            else:
                second_index = min(c2, c3)
                if second_index == c2:
                    cycles.append(set(s1[c1]) | set(s2[c2]))
                    c1 += 1
                    c2 += 1
                    current_cycle += 1
                    continue
                else:
                    cycles.append(set(s1[c1]) | set(s3[c3]))
                    c1 += 1
                    c3 += 1
                    current_cycle += 1
                    continue
        if first_index == c2:
            if s1[c1] == s2[c2] or s3[c3] == s2[c2]:
                cycles.append(set(s2[c2]) | set(s1[c1]) | set(s3[c3]))
                c1 += 1
                c2 += 1
                c3 += 1
                current_cycle += 1
                continue
            else:
                second_index = min(c1, c3)
                if second_index == c1:
                    cycles.append(set(s2[c2]) | set(s1[c1]))
                    c2 += 1
                    c1 += 1
                    current_cycle += 1
                    continue
                else:
                    cycles.append(set(s2[c2]) | set(s3[c3]))
                    c2 += 1
                    c3 += 1
                    current_cycle += 1
                    continue
        if first_index == c3:
            if s2[c2] == s3[c3] or s1[c1] == s3[c3]:
                cycles.append(set(s3[c3]) | set(s2[c2]) | set(s1[c1]))
                # if len(cycles[current_cycle]) == 1:
                #     cycles[current_cycle].add(random.choice(['A', 'C', 'G', 'T']))
                c1 += 1
                c2 += 1
                c3 += 1
                current_cycle += 1
                continue
            else:
                second_index = min(c2, c1)
                if second_index == c2:
                    cycles.append(set(s3[c3]) | set(s2[c2]))
                    c3 += 1
                    c2 += 1
                    current_cycle += 1
                    continue
                else:
                    cycles.append(set(s3[c3]) | set(s1[c1]))
                    c1 += 1
                    c3 += 1
                    current_cycle += 1
                    continue
    while c1 < n or c2 < n or c3 < n:
        cycles.append(set())
        if c1 < n:
            cycles[current_cycle].add(s1[c1])
            c1 += 1
        if c2 < n:
            cycles[current_cycle].add(s2[c2])
            c2 += 1
        if c3 < n:
            cycles[current_cycle].add(s3[c3])
            c3 += 1
        current_cycle += 1
    return cycles


def update_min_vars(arr2, c1, c2, c3):
    if arr2[0] == c1-1:
        arr2[0] += 1
    elif arr2[0] == c2-1:
        arr2[0] += 1
    elif arr2[0] == c3-1:
        arr2[0] += 1
    if arr2[1] == c1 - 1:
        arr2[1] += 1
    elif arr2[1] == c2 - 1:
        arr2[1] += 1
    elif arr2[1] == c3 - 1:
        arr2[1] += 1
    return


def generate_strings():
    chars = ['A', 'C', 'G', 'T']
    s1 = ''.join(random.choices(chars, k=10))
    s2 = ''.join(random.choices(chars, k=10))
    s3 = ''.join(random.choices(chars, k=10))
    return s1, s2, s3



if __name__ == '__main__':
    # s1 = "CTATCCTGTT"
    # s2 = "TATGCTTTCC"
    # s3 = "CCTTTCCGCA"
    i = 0
    late_counter_of_1 = 0
    early_counter_of_2 = 0
    it_matters = 0
    doesnt_matter = 0
    number_of_cycles = 0
    worst_number_of_cycles = 0
    number_of_worst = 0 # number of times worst number of cycles was received
    # worst_number_of_cycles_type = 0 # 0 means doesnt matter, 1 means late is worse, 2 means early is worse
    number_triplets = 100000
    while i < number_triplets:
        s1, s2, s3 = generate_strings()
        # print("iteration", i+1)
        # print(s1, s2, s3)
        x1_late = create_sets(s1, s2, s3, 1)
        x2_early = create_sets(s1, s2, s3, 2)
        if len(x2_early) < len(x1_late):
            number_of_cycles += len(x2_early)
            early_counter_of_2 += 1
            it_matters += 1
        elif len(x2_early) > len(x1_late):
            number_of_cycles += len(x1_late)
            late_counter_of_1 += 1
            it_matters += 1
        else:
            number_of_cycles += len(x1_late)
            doesnt_matter += 1
        if len(x1_late) > worst_number_of_cycles:
            worst_number_of_cycles = len(x1_late)
            worst_number_of_cycles_type = 1
            number_of_worst = 1
        elif len(x2_early) > worst_number_of_cycles:
            worst_number_of_cycles = len(x2_early)
            worst_number_of_cycles_type = 2
            number_of_worst = 1
        elif len(x1_late) == worst_number_of_cycles or len(x2_early) == worst_number_of_cycles:
            number_of_worst += 1
        i += 1
    print("the average number of cycles is:", number_of_cycles/number_triplets)
    print("the number of cycles it mattered:", it_matters)
    print("early matters:", early_counter_of_2)
    print("late matters:", late_counter_of_1)
    print("the number of cycles it didn't:", doesnt_matter)
    print("worst number of cycles:", worst_number_of_cycles)
    # print("its type:", worst_number_of_cycles_type)
    print("number of times worst number of cycles was received", number_of_worst)