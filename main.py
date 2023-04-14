# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import itertools
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import log, ceil


import random

def generate_random_strings(n):
    choices = ['A', 'C', 'G', 'T']
    s1 = ''.join(random.choices(choices, k=n))
    s2 = ''.join(random.choices(choices, k=n))
    s3 = ''.join(random.choices(choices, k=n))
    return s1, s2, s3


def to_base4(num, desired_length):
    """
    Given an integer num, returns its base-4 representation using the mapping A=0, C=1, G=2, T=3
    """
    digits = []
    while num > 0:
        num, remainder = divmod(num, 4)
        digits.append(remainder)
    digits.reverse()
    index = ''.join(['ACGT'[d] for d in digits]) if digits else 'A'
    diff = desired_length - len(index)
    for i in range(diff):
        index = 'A' + index
    return index


def SCSlength(X, Y, m, n):
    dp = [[0] * (n + 2) for i in range(m + 2)]
    # Fill table in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # Below steps follow above recurrence
            if (not i):
                dp[i][j] = j
            elif (not j):
                dp[i][j] = i

            elif (X[i - 1] == Y[j - 1]):
                dp[i][j] = 1 + dp[i - 1][j - 1]

            else:
                dp[i][j] = 1 + min(dp[i - 1][j],
                                   dp[i][j - 1])

    return dp[m][n]

def SCS(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize the dynamic programming table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = i + j
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
    # Backtrack to construct the shortest common supersequence
    scs = ""
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            scs = s1[i - 1] + scs
            i -= 1
            j -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            scs = s1[i - 1] + scs
            i -= 1
        else:
            scs = s2[j - 1] + scs
            j -= 1
    while i > 0:
        scs = s1[i - 1] + scs
        i -= 1
    while j > 0:
        scs = s2[j - 1] + scs
        j -= 1
    return scs

def ShortestSCSof2OutOf3(s1, s2, s3):
    s12 = SCS(s1, s2)
    s13 = SCS(s1, s3)
    s23 = SCS(s2, s3)
    if len(s12) < len(s13): #scs(1,2) < scs(1,3)
        if len(s12) < len(s23): #scs(1,2) < scs(2,3) <> scs(1,3)
            return s12, 1, 2
        else: # scs(2,3) < scs(1,2) < scs(1,3)
            return s23, 2, 3
    else: #scs(1,3) <= scs(1,2)
        if len(s13) < len(s23): # scs(1,3) < scs(2,3) <> scs(1,3)
            return s13, 1, 3
        else: #scs(2,3) < scs(1,3) <> scs(1,2)
            return s23, 2, 3


def overlap(a, b, c):
    """
    Computes the length of the longest common suffix of a, b, and c.
    """
    n = min(len(a), len(b), len(c))
    for i in range(n, 0, -1):
        if a[-i:] == b[:i] and b[-i:] == c[:i]:
            return i
    return 0


def lcs(str1, str2):
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


def lcs_with_indices(str1, str2):
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

    # Traverse the matrix to construct the LCS string and its indices
    lcs_string = ""
    s1_indices = []
    s2_indices = []
    i = m
    j = n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs_string = str1[i - 1] + lcs_string
            s1_indices.insert(0, i - 1)
            s2_indices.insert(0, j - 1)
            i -= 1
            j -= 1
        elif lcs_matrix[i - 1][j] > lcs_matrix[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return s1_indices, s2_indices


def lcs_of_3_strings(str1, str2, str3): #returns lcs12, lcs13, lcs23

    lcs_dict={}
    # Compute LCS of str1 and str2
    lcs_dict['1in12'], lcs_dict['2in12'] = lcs_with_indices(str1, str2)

    # Compute LCS of str1 and str3
    lcs_dict['1in13'], lcs_dict['3in13'] = lcs_with_indices(str1, str3)

    # Compute LCS of str2 and str3
    lcs_dict['2in23'], lcs_dict['3in23'] = lcs_with_indices(str2, str3)


    return lcs_dict


# def scs(a, b, c):
#     """
#     Returns the shortest common superstring of three strings.
#     """
#     n = len(a)
#     m = len(b)
#     l = len(c)
#     dp = [[[0 for _ in range(l+1)] for _ in range(m+1)] for _ in range(n+1)]
#
#     for i in range(n+1):
#         for j in range(m+1):
#             for k in range(l+1):
#                 if i == 0 or j == 0 or k == 0:
#                     dp[i][j][k] = 0
#                 elif a[i-1] == b[j-1] == c[k-1]:
#                     dp[i][j][k] = dp[i-1][j-1][k-1] + 1
#                 else:
#                     dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1], dp[i-1][j-1][k], dp[i-1][j][k-1], dp[i][j-1][k-1], overlap(a[:i], b[:j], c[:k]))
#
#     i = n
#     j = m
#     k = l
#     result = []
#     while i > 0 or j > 0 or k > 0:
#         if i == 0 or j == 0 or k == 0:
#             break
#         if dp[i][j][k] == dp[i-1][j][k]:
#             i -= 1
#         elif dp[i][j][k] == dp[i][j-1][k]:
#             j -= 1
#         elif dp[i][j][k] == dp[i][j][k-1]:
#             k -= 1
#         elif a[i-1] == b[j-1] == c[k-1]:
#             result.append(a[i-1])
#             i -= 1
#             j -= 1
#             k -= 1
#         elif dp[i][j][k] == dp[i-1][j-1][k]:
#             result.append(a[i-1])
#             i -= 1
#             j -= 1
#         elif dp[i][j][k] == dp[i-1][j][k-1]:
#             result.append(a[i-1])
#             i -= 1
#             k -= 1
#         elif dp[i][j][k] == dp[i][j-1][k-1]:
#             result.append(b[j-1])
#             j -= 1
#             k -= 1
#
#     return ''.join(result[::-1]) + a[i:n] + b[j:m] + c[k:l]
#
#
#     result.reverse()
#     return "".join(result)


# Example usage:

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

def create_sets(cycles, s1, s2, s3):
    strings = s1, s2, s3
    n = max(len(s) for s in strings)
    result = []
    for i in range(n):
        letters = cycles[i]
        for s in strings:
            if i < len(s):
                letters.add(s[i])
        while len(letters) == 1:
            letters.add(random.choice(['A', 'C', 'G', 'T']))
        result.append(letters)


def GetCycles1(s1, s2, s3): # returns the number of cycles needed when addressing the common indices of at least two strands,
    # and calculates the SCS of the rest of the strings
    cycles = [set() for _ in range(2 * len(s1))]
    different_letters_on_each_index = count_different_letters(s1, s2, s3)
    print(different_letters_on_each_index)
    if all(val <= 2 for val in different_letters_on_each_index):  # if we can synthesize the 3 strings in n cycles
        create_sets(cycles, s1, s2, s3)
        cycles = [s for s in cycles if len(s) > 0]
        print(cycles)  # finish here, need to return the cycles
        return cycles
    else:
        for i in range(len(different_letters_on_each_index)):
            if different_letters_on_each_index[i] <= 2:
                cycles[i] = set({s1[i], s2[i], s3[i]})
                while len(cycles[i]) == 1:
                    cycles[i].add(random.choice(['A', 'C', 'G', 'T']))
        print(cycles)


def lcs_three(s1, s2, s3):
    # Initialize the memoization table
    m, n, l = len(s1), len(s2), len(s3)
    dp = [[[0 for _ in range(l + 1)] for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill the memoization table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            for k in range(1, l + 1):
                if s1[i - 1] == s2[j - 1] == s3[k - 1]:
                    dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1
                else:
                    dp[i][j][k] = max(dp[i - 1][j][k], dp[i][j - 1][k], dp[i][j][k - 1])

    # Backtrack to find the LCS
    i, j, k = m, n, l
    lcs = ""
    while i > 0 and j > 0 and k > 0:
        if s1[i - 1] == s2[j - 1] == s3[k - 1]:
            lcs = s1[i - 1] + lcs
            i -= 1
            j -= 1
            k -= 1
        elif dp[i - 1][j][k] >= dp[i][j][k] and dp[i - 1][j][k] >= dp[i][j - 1][k] and dp[i - 1][j][k] >= dp[i][j][
            k - 1]:
            i -= 1
        elif dp[i][j - 1][k] >= dp[i][j][k] and dp[i][j - 1][k] >= dp[i - 1][j][k] and dp[i][j - 1][k] >= dp[i][j][
            k - 1]:
            j -= 1
        elif dp[i][j][k - 1] >= dp[i][j][k] and dp[i][j][k - 1] >= dp[i][j - 1][k] and dp[i][j][k - 1] >= dp[i - 1][j][
            k]:
            k -= 1

    return lcs


def GetCycles2(s1, s2, s3): #returns the number of cylces needed when addressing the LCS of two
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # generate all possible combinations of three 10-length strings
    # combinations = itertools.combinations_with_replacement("ATGC", 10)
    # loop through all combinations and calculate the shortest common superstring
    # with open('newTryWnoam.txt', mode='w') as file:
        # for s1, s2, s3 in combinations:
        #     result = shortest_common_superstring(s1, s2, s3)
        #     print(f"{s1} , {s2} , {s3} -----> {result}", file=file)
    # s1 = "AAAAA"
    # s2 = "AACAA"
    # s3 = "ACTTAT"
    # s12 = shortest_common_superstring(s1, s2)
    # s12s3 = shortest_common_superstring(s12, s3)
    # s3s12 = shortest_common_superstring(s3, s12)
    # s21 = shortest_common_superstring(s2, s1)
    # s21s3 = shortest_common_superstring(s21, s3)
    # s3s21 = shortest_common_superstring(s3, s21)
    # s13 = shortest_common_superstring(s1, s3)
    # s13s2 = shortest_common_superstring(s13, s2)
    # s2s13 = shortest_common_superstring(s2, s13)
    # s31 = shortest_common_superstring(s3, s1)
    # s31s2 = shortest_common_superstring(s31, s2)
    # s2s31 = shortest_common_superstring(s2, s31)
    # s23 = shortest_common_superstring(s2, s3)
    # s1s23 = shortest_common_superstring(s1, s23)
    # s23s1 = shortest_common_superstring(s23, s1)
    # s32 = shortest_common_superstring(s3, s2)
    # s1s32 = shortest_common_superstring(s1, s32)
    # s32s1 = shortest_common_superstring(s32, s1)
    # freq_dict = {}
    # for i in range(1, 4):
    #     for j in range(1, 4):
    #         if i == j:
    #             continue
    #         for k in range(1, 4):
    #             if k == j or k == i:
    #                 continue
    #             cur1 = 's' + str(i) + 's' + str(j) + str(k)
    #             cur2 = 's' + str(j) + str(k) + 's' + str(i)
    #             val1 = locals()[cur1]
    #             val2 = locals()[cur2]
    #             print(cur1 + " = " + val1)
    #             print(cur2 + " = " + val2)
    #             if val1 not in freq_dict:
    #                 freq_dict[val1] = 1
    #             else:
    #                 freq_dict[val1] += 1
    #             if val2 not in freq_dict:
    #                 freq_dict[val2] = 1
    #             else:
    #                 freq_dict[val2] += 1
    # plt.hist(freq_dict.values())
    # plt.xlabel('Frequency')
    # plt.ylabel('Count')
    # plt.title('Histogram of Variable Values')
    # plt.show()
    # s1, s2, s3 = generate_random_strings(5)
    n = 5
    s1 = "GAGAGA"
    s2 = "GGGGGG"
    s3 = "GAGAGA"
    print(lcs_three(s1, s2, s3))
    # print(s2)
    # print(s3)
    # print(real_scs)
    # GetCycles1(s1, s2, s3)




    # cycles = [set() for _ in range(2*n)]
    # different_letters_on_each_index = count_different_letters(s1, s2, s3)
    # if all(val <= 2 for val in different_letters_on_each_index): #if we can synthesize the 3 strings in n cycles
    #     cycles = create_sets(s1, s2, s3)
    #     print(cycles) #finish here, need to return the cycles
    # n = len(s1)
    # # lcs_dict = lcs_of_3_strings(s1, s2, s3)
    # # print(scs(s1, s2, s3))
    # # lcs12 = len(lcs_of_pairs[0])
    # # lcs13 = len(lcs_of_pairs[2])
    # # lcs23 = len(lcs_of_pairs[4])
    # # print(str(lcs12) + " = " + str(lcs_of_pairs[0]) + " " + str(lcs_of_pairs[1]))
    # # print(str(lcs13) + " = " + str(lcs_of_pairs[2]) + " " + str(lcs_of_pairs[3]))
    # # print(str(lcs23) + " = " + str(lcs_of_pairs[4]) + " " + str(lcs_of_pairs[4]))
    # # ind1 = set(lcs_of_pairs[0]) | set((lcs_of_pairs[2]))
    # # print(ind1)
    # # ind2 = set(lcs_of_pairs[1])| set(lcs_of_pairs[4])
    # # print(ind2)
    # # ind3 = set(lcs_of_pairs[3])|set(lcs_of_pairs[5])
    # # print(ind3)
    # #look for the maximum length lcs
    # # generate the scs of the two
    # #LCS SOL
    # # all_indices = set(range(0, n))
    # # ind1 = set(lcs_dict['1in12'] + lcs_dict['1in13']) #indices of s1 that are common with the other strings
    # # ind2 = set(lcs_dict['2in12'] + lcs_dict['2in23']) #indices of s2 that are common with the other strings
    # # ind3 = set(lcs_dict['3in13'] + lcs_dict['3in23']) #indices of s3 that are common with the other strings
    # # remain1 = all_indices - ind1 #indices of s1 that are unique to it when compared to the lcs
    # # remain2 = all_indices - ind2 #indices of s2 that are unique to it when compared to the lcs
    # # remain3 = all_indices - ind3 #indices of s3 that are unique to it when compared to the lcs
    # ss, first, second = ShortestSCSof2OutOf3(s1, s2, s3)
    # extra_string = ""
    # if first != 1: # ss = scs(2,3)
    #     create_sets(cycles, ss, ss, s1)
    #     extra_string = s1
    # elif second == 2: # ss = scs(1,2)
    #     create_sets(cycles, ss, ss, s3)
    #     extra_string = 23
    # else: # ss = scs(1,3)
    #     create_sets(cycles, ss, ss, s2)
    #     extra_string = s2
    # print(ss)
    # print(extra_string)
    # print("The number of cycles needed is " + str(len(cycles)))
    # print(cycles)






    # ss_len = len(ss)
    # in1 = ss.find(s1)
    # in2 = ss.find(s2)
    # in3 = ss.find(s3)
    # print("the shortest superstring of " + s1 + ", " + s2 + ",and " + s3 + " is " + ss + ", and its length is " + str(ss_len))
    # print("indexes are: " + str(in1) + ", " + str(in2) + ", and " + str(in3))
    # print(to_base4(in1, 3))
    # print(to_base4(in2, 3))
    # print(to_base4(in3, 3))
    # ss = to_base4(ss.find(s1), 3) + to_base4(ss.find(s2), 3) + to_base4(ss.find(s3), 3) + ss
    # print(ss)

    # print("the shortest superstring of " + s1 + ", " + s2 + ",and " + s3 + " is " + ss + ", and its length is " + str(ss_len))
    # print(to_base4())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/



