import random
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# from collections import defaultdict


# for the sake of code clarity, there is not a real need in python to use a swap function
def swap(a, b):
    return b, a


def lcs_with_low_indexes(s1, s2):  # returns the lcs with the lowest indexes, and its indexes
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


def high_indexes_lcs(str1, str2):  # returns the lcs with the highest indexes, and its indexes
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


# returns indexes of lcs, highest indexes
def indexes_of_high_indexes_lcs(str1, str2):
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


def leading_indexes1(str1, str2,
                     str3):  # maps strings to lcs indexes# (highest indexes)
    # Compute indexes of LCS of every pair
    lcs_dict = {'1in12': (indexes_of_high_indexes_lcs(str1, str2))[0], '2in12': (indexes_of_high_indexes_lcs(str1, str2))[1],
                '1in13': (indexes_of_high_indexes_lcs(str1, str3))[0], '3in13': (indexes_of_high_indexes_lcs(str1, str3))[1],
                '2in23': (indexes_of_high_indexes_lcs(str2, str3))[0], '3in23': (indexes_of_high_indexes_lcs(str2, str3))[1]}
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
    lcs12, lcs_dict['1in12'], lcs_dict['2in12'] = lcs_with_low_indexes(str1, str2)
    # Compute indexes of LCS of str1 and str3
    lcs13, lcs_dict['1in13'], lcs_dict['3in13'] = lcs_with_low_indexes(str1, str3)
    # Compute indexes of LCS of str2 and str3
    lcs23, lcs_dict['2in23'], lcs_dict['3in23'] = lcs_with_low_indexes(str2, str3)
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


def create_synthesis_seq(string1, string2, string3, instance):
    x = 0
    if instance == 1:
        x = leading_indexes1(s1, s2, s3)
    else: #instance == 2
        x = leading_indexes2(s1, s2, s3)
    # print(x)
    lcs_strings = [key[0] for key in x.keys()]
    lcs_indexes = [value for value in x.values()]
    lcs_len = len(lcs_indexes[0])
    lcs1_indexes = lcs_indexes[0]
    lcs2_indexes = lcs_indexes[1]
    a = int(lcs_strings[0])
    b = int(lcs_strings[1])
    if a != 1:  # s23
        string1, string2 = swap(string1, string2)
        string2, string3 = swap(string2, string3)
    elif b == 3:
        string2, string3 = swap(string2, string3)
    # the lcs is of s1 and s2, and the indexes of the lcs in s1 is lcs1_indexes,
    # and the indexes of the lcs in s2 is lcs2_indexes
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
    lcs_arr = [s1[i] for i in lcs1_indexes]
    lcs = ''.join(lcs_arr)
    return s1, s2, s3, cycles, lcs_len, lcs


def update_min_vars(arr2, c1, c2, c3):
    if arr2[0] == c1 - 1:
        arr2[0] += 1
    elif arr2[0] == c2 - 1:
        arr2[0] += 1
    elif arr2[0] == c3 - 1:
        arr2[0] += 1
    if arr2[1] == c1 - 1:
        arr2[1] += 1
    elif arr2[1] == c2 - 1:
        arr2[1] += 1
    elif arr2[1] == c3 - 1:
        arr2[1] += 1
    return


def generate_strings(n):
    chars = ['A', 'C', 'G', 'T']
    s1 = ''.join(random.choices(chars, k=n))
    s2 = ''.join(random.choices(chars, k=n))
    s3 = ''.join(random.choices(chars, k=n))
    return s1, s2, s3


def naive(str1, str2, str3):
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
        else:
            # in the current cycle there are three different letters
            random_chars = random.sample(letters_of_cycle, k=2)
            cycles.append(random_chars)
            if s1[c1] in random_chars:
                c1 += 1
            if s2[c2] in random_chars:
                c2 += 1
            if s3[c3] in random_chars:
                c3 += 1
            current_cycle += 1
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

def triple_edit_distance(str1, str2, str3):
    len1, len2, len3 = len(str1), len(str2), len(str3)
    # Initialize the distance matrix
    distance = [[[0 for _ in range(len3+1)] for _ in range(len2+1)] for _ in range(len1+1)]
    for i in range(len1+1):
        for j in range(len2+1):
            for k in range(len3+1):
                if i == 0:
                    distance[i][j][k] = j + k
                elif j == 0:
                    distance[i][j][k] = i + k
                elif k == 0:
                    distance[i][j][k] = i + j
                else:
                    cost = 0 if (str1[i-1] == str2[j-1] == str3[k-1]) else 1
                    distance[i][j][k] = min(distance[i-1][j-1][k-1]+cost,
                                            distance[i-1][j][k]+1,
                                            distance[i][j-1][k]+1,
                                            distance[i][j][k-1]+1)
    return distance[len1][len2][len3]


def edit_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

def plot_dict(dictionary):
    fig, ax = plt.subplots()
    ax.bar(dictionary.keys(), dictionary.values())
    ax.set_xlabel("Keys")
    ax.set_ylabel("Values")
    ax.set_title("Dictionary plot")
    plt.show()


if __name__ == '__main__':
    triplet_counter = 0
    counter1 = 0    # counter of x1_high
    counter2 = 0    # counter of x2_low
    it_matters = 0
    doesnt_matter = 0
    number_of_cycles = 0
    worst_number_of_cycles = 0
    number_of_worst = 0  # number of times worst number of cycles was received
    # worst_number_of_cycles_type = 0 # 0 means doesnt matter, 1 means high is worse, 2 means low is worse
    number_of_triplets = 100
    synthesis_times = {}  # dictionary to store the times of synthesis per set
    n = 10
    naive_avg = 0
    edit_dist_of_triplet = {}
    # Create two lists to store the difference between synthesis_times and upper_bound
    set_lengths = []
    upper_bound = 0
    data = {
        'index': [],
        'syn_time': [],
        'lcs_length': [],
        'upper_bound': [],
        'upper_bound-syn_time': [],
        'edit_dist_of_all_three': [],
        'edit_dist_of_the_lcs_couple': [],
        'edit_dist2': [],
        'edit_dist3': [],
        'naive_syn_time': []
    }
    worst_syn_time_sets = {
        'index': [],
        's1': [],
        's2': [],
        's3': [],
        'syn_time': [],
        'lcs': [],
        'lcs_length': [],
        'upper_bound': [],
        'upper_bound-syn_time': [],
        'edit_dist_of_all_three': [],
        'edit_dist_of_the_lcs_couple': [],
        'edit_dist2': [],
        'edit_dist3': [],
        'naive_syn_time': []
    }
    worse_than_naive = {
        'index': [],
        's1': [],
        's2': [],
        's3': [],
        'syn_time': [],
        'lcs': [],
        'lcs_length': [],
        'upper_bound': [],
        'upper_bound-syn_time': [],
        'edit_dist_of_all_three': [],
        'edit_dist_of_the_lcs_couple': [],
        'edit_dist2': [],
        'edit_dist3': [],
        'naive_syn_time': []
    }

    while triplet_counter < number_of_triplets:
        data['index'].append(triplet_counter)
        s1, s2, s3 = generate_strings(n)
        # print("iteration", i+1)
        data['edit_dist_of_all_three'].append(triple_edit_distance(s1, s2, s3))
        s1_1, s2_1, s3_1, x1_high, lcs_len1, lcs_1 = create_synthesis_seq(s1, s2, s3, 1)
        s1_2, s2_2, s3_2, x2_low, lcs_len2, lcs_2 = create_synthesis_seq(s1, s2, s3, 2)
        if lcs_len1 < lcs_len2:
            s1, s2, s3 = s1_1, s2_1, s3_1
            data['lcs_length'].append(lcs_len1)
        else:
            s1, s2, s3 = s1_2, s2_2, s3_2
            data['lcs_length'].append(lcs_len2)
        # s1 and s2 are the lcs strings
        data['edit_dist_of_the_lcs_couple'].append(edit_distance(s1, s2))
        data['edit_dist2'].append(edit_distance(s1, s3))
        data['edit_dist3'].append(edit_distance(s2, s3))
        x1_len = len(x1_high)
        x2_len = len(x2_low)
        naive_pick = naive(s1, s2, s3)
        naive_pick_len = len(naive_pick)
        data['naive_syn_time'].append(naive_pick_len)
        naive_avg += naive_pick_len
        data['syn_time'].append(min(x2_len,x1_len))
        if x2_len < x1_len:
            number_of_cycles += x2_len
            counter2 += 1
            it_matters += 1
            if x2_len in synthesis_times:
                synthesis_times[x2_len] += 1
            else:
                synthesis_times[x2_len] = 1
            upper_bound = lcs_len2 + (n-lcs_len2)*2
            diff = upper_bound - x2_len
            data['upper_bound'].append(upper_bound)
            data['upper_bound-syn_time'].append(diff)
            if x2_len > worst_number_of_cycles:
                worst_number_of_cycles = x2_len
                worst_number_of_cycles_type = 2
                number_of_worst = 1
            elif x2_len == worst_number_of_cycles:
                number_of_worst += 1
        elif x2_len > x1_len:
            number_of_cycles += x1_len
            counter1 += 1
            it_matters += 1
            upper_bound = lcs_len1 + (n - lcs_len1) * 2
            diff = upper_bound - x1_len
            data['upper_bound'].append(upper_bound)
            data['upper_bound-syn_time'].append(diff)
            if x1_len > worst_number_of_cycles:
                worst_number_of_cycles = x1_len
                worst_number_of_cycles_type = 1
                number_of_worst = 1
            elif x1_len == worst_number_of_cycles:
                number_of_worst += 1
        else:
            number_of_cycles += x1_len
            doesnt_matter += 1
            upper_bound = lcs_len1 + (n - lcs_len1) * 2
            if x1_len in synthesis_times:
                synthesis_times[x1_len] += 1
            else:
                synthesis_times[x1_len] = 1
            upper_bound = lcs_len1 + (n - lcs_len1) * 2
            diff = upper_bound - x1_len
            data['upper_bound'].append(upper_bound)
            data['upper_bound-syn_time'].append(diff)
            if x1_len > worst_number_of_cycles:
                worst_number_of_cycles = x1_len
                worst_number_of_cycles_type = 1
                number_of_worst = 1
            elif x1_len == worst_number_of_cycles:
                number_of_worst += 1
        #here the data of the current set is filled
        #insert to worst_syn_time_sets if needed
        if len(set(worst_syn_time_sets['s1'])) < 3 or (data['syn_time'])[triplet_counter] >= min(worst_syn_time_sets['syn_time']):
            worst_syn_time_sets['s1'].append(s1)
            worst_syn_time_sets['s2'].append(s2)
            worst_syn_time_sets['s3'].append(s3)
            if x1_len < x2_len:
                worst_syn_time_sets['lcs'].append(lcs_1)
            else:
                worst_syn_time_sets['lcs'].append(lcs_2)
            for lable_in_data in data.keys():
                worst_syn_time_sets[lable_in_data].append(data[lable_in_data][triplet_counter])
            if len(set(worst_syn_time_sets['syn_time'])) > 3:
                min_index = (worst_syn_time_sets['syn_time']).index(min(worst_syn_time_sets['syn_time']))
                for lable_in_worst in worst_syn_time_sets.keys():
                    worst_syn_time_sets[lable_in_worst].pop(min_index)
        # insert to worse_than_naive if needed
        if data['syn_time'][triplet_counter] > data['naive_syn_time'][triplet_counter]:
            worse_than_naive['s1'].append(s1)
            worse_than_naive['s2'].append(s2)
            worse_than_naive['s3'].append(s3)
            if x1_len < x2_len:
                worse_than_naive['lcs'].append(lcs_1)
            else:
                worse_than_naive['lcs'].append(lcs_2)
            for lable_in_data in data.keys():
                worse_than_naive[lable_in_data].append(data[lable_in_data][triplet_counter])
        triplet_counter += 1

    total_final_syn_time = []
    total_upper_bounds_difference = []
    for i in range(triplet_counter):
        if data['index'][i] not in worse_than_naive['index']:
            total_final_syn_time.append(data['syn_time'][i])
        else:
            total_final_syn_time.append(data['naive_syn_time'][i])
        total_upper_bounds_difference.append(data['upper_bound'][i] - total_final_syn_time[i])
    avg_final_syn_time = sum(total_final_syn_time)/triplet_counter


    print("the average number of cycles, without minimizing with naive, is:", number_of_cycles / number_of_triplets)
    print("the final average number of cycle is:", sum(total_final_syn_time) / number_of_triplets)
    print("average edit distance of three where the naive is better: ", sum(worse_than_naive['edit_dist_of_all_three'])/len(worse_than_naive['edit_dist_of_all_three']))
    print("average edit distance of the LCS-couple where the naive is better: ", sum(worse_than_naive['edit_dist_of_the_lcs_couple'])/len(worse_than_naive['edit_dist_of_the_lcs_couple']))
    print("average lcs-length: ", sum(data['lcs_length'])/len(data['lcs_length']))
    print("average lcs-length where the naive is better: ", sum(worse_than_naive['lcs_length'])/len(worse_than_naive['lcs_length']))
    print("average lcs-length of the worst synthesis-time sets: ", sum(worst_syn_time_sets['lcs_length']) / len(worst_syn_time_sets['lcs_length']))
    print("the number of cycles it mattered:", it_matters)
    print("low matters:", counter2)
    print("high matters:", counter1)
    print("the number of cycles it didn't:", doesnt_matter)
    print("worst number of cycles:", worst_number_of_cycles)
    # print("its type:", worst_number_of_cycles_type)
    print("number of times worst number of cycles was received", number_of_worst)
    print("the average number of cycles by naive is:", naive_avg/number_of_triplets)
    print("the average difference between the upper bound and our syn time is:", sum(total_upper_bounds_difference) / len(total_upper_bounds_difference))
    print(synthesis_times.keys())
    print(synthesis_times.values())

    worst_filename = "WorstSynTimeOf" + str(triplet_counter) + "SetsLength" + str(n)
    worse_than_naive_filename = "WorseThanNaiveOf" + str(triplet_counter) + "SetsLength" + str(n)
    write_worst = True
    write_worse = True
    iterations = max(len(worst_syn_time_sets['index']), len(worse_than_naive['index']))
    with open(worst_filename, "w") as f1:
        with open(worse_than_naive_filename, "w") as f2:
            for i in range(iterations):
                f1.write(str(i) + ":\n")
                if i < len(worst_syn_time_sets['index']):
                    for lbl1 in worst_syn_time_sets.keys():
                        f1.write(lbl1 + ":    " + str(worst_syn_time_sets[lbl1][i]) + "\n")
                else:
                    write_worst = False
                f2.write(str(i) + ":\n")
                if i < len(worse_than_naive['index']):
                    for lbl2 in worse_than_naive.keys():
                        f2.write(lbl2 + ":    " + str(worse_than_naive[lbl2][i]) + "\n")
                else:
                    write_worse = False
                if write_worst is False and write_worse is False:
                    break
    # plot the histogram of synthesis times
    plt.bar(synthesis_times.keys(), synthesis_times.values())
    plt.xlabel('Synthesis Times')
    plt.ylabel('Number of Triplets Synthesized')
    plt.title('Distribution of set lengths')
    plt.show()

    #plot the differences(bound - Sythesis_time)  per synthesis time
    data_sorted = sorted(data['upper_bound-syn_time'])
    # Get the counts of each unique value in the list and sort by value
    counts = pd.Series(data_sorted).value_counts().sort_index()
    # Create a bar chart of the counts
    ax = counts.plot.bar()
    # Add axis labels and title to the plot
    ax.set_xlabel('Difference Between Upper Bound and the Synthesis Time')
    ax.set_ylabel('Number of Triplets')
    ax.set_title('Difference Counts')
    # Display the plot
    plt.show()

    # plot the number of generated strings, by LCS index type
    labels = ['Highest Indexed LCS', 'Lowest Indexed LCS', "Doesn't Matter"]
    values = [counter1, counter2, doesnt_matter]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.xlabel('String type')
    plt.ylabel('Number of Generated Strings')
    plt.title('Number of Generated Strings by LCS Indexing')
    plt.show()

    # Create a defaultdict to count the number of occurrences of each point
    #for each synthesis time - what is the length of the lcs
    # create a pandas dataframe from the two lists
    df = pd.DataFrame({'syn_time': data['syn_time'], 'lcs_length': data['lcs_length']})
    # calculate the counts for each combination of syn_time and lcs_length
    counts = df.groupby(['syn_time', 'lcs_length']).size().reset_index(name='count')
    # create a pivot table with syn_time as columns and lcs_length as rows
    pivot_table = counts.pivot(index='lcs_length', columns='syn_time', values='count')
    # create the heatmap using seaborn
    ax = sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='g')
    # reverse the order of the y-axis labels
    ax.set_yticklabels(reversed(ax.get_yticklabels()))
    plt.title('Synthesis Time and the LCS Length')
    plt.show()
    # counts = defaultdict(int)

    # create a pandas dataframe from the two lists
    df = pd.DataFrame({'Synthesis Time': data['syn_time'], 'Edit Distance': data['edit_dist_of_all_three']})
    # calculate the counts for each combination of syn_time and lcs_length
    counts = df.groupby(['Synthesis Time', 'Edit Distance']).size().reset_index(name='count')
    # create a pivot table with syn_time as columns and lcs_length as rows
    pivot_table = counts.pivot(index='Edit Distance', columns='Synthesis Time', values='count')
    # create the heatmap using seaborn
    ax = sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='g')
    # reverse the order of the y-axis labels
    ax.set_yticklabels(reversed(ax.get_yticklabels()))
    plt.title('Synthesis Time and the Edit Distance of the Three Strands')
    plt.show()

    # create a pandas dataframe from the two lists
    df = pd.DataFrame({'Synthesis Time': data['syn_time'], 'Edit Distance': data['edit_dist_of_the_lcs_couple']})
    # calculate the counts for each combination of syn_time and lcs_length
    counts = df.groupby(['Synthesis Time', 'Edit Distance']).size().reset_index(name='count')
    # create a pivot table with syn_time as columns and lcs_length as rows
    pivot_table = counts.pivot(index='Edit Distance', columns='Synthesis Time', values='count')
    # create the heatmap using seaborn
    ax = sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='g')
    # reverse the order of the y-axis labels
    ax.set_yticklabels(reversed(ax.get_yticklabels()))
    plt.title('Synthesis Time and the Edit Distance of the LCS couple')
    plt.show()


    # create a pandas dataframe from the two lists
    df = pd.DataFrame({'Synthesis Time': data['syn_time'], 'Edit Distance': data['edit_dist2']})
    # calculate the counts for each combination of syn_time and lcs_length
    counts = df.groupby(['Synthesis Time', 'Edit Distance']).size().reset_index(name='count')
    # create a pivot table with syn_time as columns and lcs_length as rows
    pivot_table = counts.pivot(index='Edit Distance', columns='Synthesis Time', values='count')
    # create the heatmap using seaborn
    ax = sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='g')
    # reverse the order of the y-axis labels
    ax.set_yticklabels(reversed(ax.get_yticklabels()))
    plt.title('Synthesis Time and the Edit Distance\nof the non-LCS Strand with One of the LCS Strands')
    plt.show()


    # create a pandas dataframe from the two lists
    df = pd.DataFrame({'Synthesis Time': data['syn_time'], 'Edit Distance': data['edit_dist3']})
    # calculate the counts for each combination of syn_time and lcs_length
    counts = df.groupby(['Synthesis Time', 'Edit Distance']).size().reset_index(name='count')
    # create a pivot table with syn_time as columns and lcs_length as rows
    pivot_table = counts.pivot(index='Edit Distance', columns='Synthesis Time', values='count')
    # create the heatmap using seaborn
    ax = sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='g')
    # reverse the order of the y-axis labels
    ax.set_yticklabels(reversed(ax.get_yticklabels()))
    plt.title('Synthesis Time and the Edit Distance\nof the non-LCS Strand with the other LCS-Strand')
    plt.show()


    # Create a figure and axis object
    fig, ax = plt.subplots()
    # Plot naive_syn_time as a red line
    ax.plot(data['naive_syn_time'], color='red', label='naive_syn_time')
    # Plot syn_time as a blue line
    ax.plot(data['syn_time'], color='blue', label='syn_time')
    # Add a legend to the plot
    ax.legend()
    differences = [naive_syn - syn for syn, naive_syn in zip(data['syn_time'], data['naive_syn_time'])]
    avg_diff = np.mean(differences)
    avg_percent_diff = avg_diff * n/100
    # Set the x and y axis labels and title
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Synthesis Time')
    ax.set_title('Comparison of Synthesis Times between\nOur Algorithm and the Naive Algorithm')
    textstr = f'Average Difference: {avg_diff:.2f}\nAverage % Difference: {avg_diff:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    # Display the plot
    plt.show()

    # import matplotlib.pyplot as plt
    #
    # # two different length lists
    # x = worse_than_naive['edit_dist_of_all_three']
    # y1 = worse_than_naive['syn_time']
    # y2 = worse_than_naive['naive_syn_time']
    #
    # # plot the line graph
    # plt.scatter(x, y1, color='red', label='Ours')
    #
    # # plot the scatter graph
    # plt.scatter(x, y2, color='blue', label='Naive')
    #
    # # add legend and labels
    # plt.legend()
    # plt.xlabel('Edit Distance')
    # plt.ylabel('Synthesis Time')
    # plt.title('Synthesis Time of the ')
    # # show the plot
    # plt.show()


    # Define your data
    edit_distance = data['edit_dist_of_all_three']
    upper_bound = data['upper_bound']
    synthesis_time = total_final_syn_time
    # Create a new figure and set its size
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the upper bound line
    ax.plot(edit_distance, upper_bound, label='Upper Bound', linestyle='None', marker='o')
    # Plot the synthesis time line
    ax.plot(edit_distance, synthesis_time, label='Our Synthesis Time', linestyle='None', marker='s')
    # Set the axis labels and title
    ax.set_xlabel('Edit Distance')
    ax.set_ylabel('Time (Cycles)')
    ax.set_title('Synthesis Time vs. Three Strands Edit Distance')
    # Add a legend to the plot
    ax.legend()
    # Show the plot
    plt.show()


    # Define your data
    edit_distance = data['edit_dist_of_the_lcs_couple']
    upper_bound = data['upper_bound']
    synthesis_time = total_final_syn_time
    # Create a new figure and set its size
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the upper bound line
    ax.plot(edit_distance, upper_bound, label='Upper Bound', linestyle='None', marker='o')
    # Plot the synthesis time line
    ax.plot(edit_distance, synthesis_time, label='Our Synthesis Time', linestyle='None', marker='s')
    # Set the axis labels and title
    ax.set_xlabel('Edit Distance')
    ax.set_ylabel('Time (Cycles)')
    ax.set_title('Synthesis Time vs. Edit Distance of the LCS Couple')
    # Add a legend to the plot
    ax.legend()
    # Show the plot
    plt.show()
