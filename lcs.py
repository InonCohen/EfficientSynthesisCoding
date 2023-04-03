import random

def swap(a, b):
    return b,a

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


def indices_of_lcs(str1, str2):
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





def leading_indices(str1, str2, str3):   # maps strings to lcs indices
    lcs_dict = {}
    # Compute indices of LCS of str1 and str2
    lcs_dict['1in12'], lcs_dict['2in12'] = indices_of_lcs(str1, str2)
    # Compute indices of LCS of str1 and str3
    lcs_dict['1in13'], lcs_dict['3in13'] = indices_of_lcs(str1, str3)
    # Compute indices of LCS of str2 and str3
    lcs_dict['2in23'], lcs_dict['3in23'] = indices_of_lcs(str2, str3)

    # create a dictionary to store the number of indices for each string's LCS
    indices_count = {
        '1in12': len(lcs_dict['1in12']),
        '2in12': len(lcs_dict['2in12']),
        '1in13': len(lcs_dict['1in13']),
        '3in13': len(lcs_dict['3in13']),
        '2in23': len(lcs_dict['2in23']),
        '3in23': len(lcs_dict['3in23'])
    }

    # sort the dictionary by the number of indices
    sorted_indices = sorted(indices_count.items(), key=lambda x: x[1], reverse=True)

    # create a new dictionary with only the top two strings and their indices
    if (sorted_indices[0][0] == '1in12' and sorted_indices[1][0] == '2in12') or \
            (sorted_indices[0][0] == '1in13' and sorted_indices[1][0] == '3in13') or \
            (sorted_indices[0][0] == '2in23' and sorted_indices[1][0] == '3in23'):
        top_two_strings = {
            sorted_indices[0][0]: lcs_dict[sorted_indices[0][0]],
            sorted_indices[1][0]: lcs_dict[sorted_indices[1][0]]
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




def create_sets(s1, s2, s3):
    x = leading_indices(s1, s2, s3)
    if x == {}:
        return separate_strings(s1, s2, s3)
    lcs_strings = [key[0] for key in x.keys()]
    lcs_indices = [value for value in x.values()]
    lcs1 = lcs_indices[0]
    lcs2 = lcs_indices[1]
    l1 = int(lcs_strings[0])
    l2 = int(lcs_strings[1])
    if l1 != 1: #s23
        s1, s2 = swap(s1,s2)
        s2, s3 = swap(s2, s3)
    elif l2 == 3:
        s2, s3 = swap(s2, s3)
    #the lcs is of s1 and s2, and the indices of the lcs in s1 is lcs1, and the indices of the lcs in s2 is lcs2
    n = len(s1)         #assume all strings are of length n
    c1, c2, c3, current_cycle = 0, 0, 0, 0    #c1 iterates over s1, c2 over s2, c3 over s3
    cycles = []
    while c1 < n and c2 < n and c3 < n:
        if s1[c1] != s2[c2] and s1[c1] != s3[c3] and s2[c2] != s3[c3]:
            if c1 in lcs1:
                if c2 in lcs2:
                    cycles.append(set(s1[c1]) | set(s2[c2]) | set(s3[c3]))
                    while len(cycles[current_cycle]) == 1:
                        cycles[current_cycle].add(random.choice(['A', 'C', 'G', 'T']))
                    c1 += 1

        first_index = min(c1, c2, c3)   #first index to be synthesized
        if first_index == c1:
            if s2[c2] == s1[c1] or s3[c3] == s1[c1]:
                cycles.append(set(s1[c1]) | set(s2[c2]) | set(s3[c3]))
                while len(cycles[current_cycle]) == 1:
                    cycles[current_cycle].add(random.choice(['A', 'C', 'G', 'T']))
                c1 += 1
                c2 += 1
                c3 += 1
                current_cycle += 1
                continue
            else:
                second_index = min(c2, c3)
                if second_index == c2:
                    cycles.append(set(s1[c1]) | set(s2[c2]) )
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
                while len(cycles[current_cycle]) == 1:
                    cycles[current_cycle].add(random.choice(['A', 'C', 'G', 'T']))
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
                while len(cycles[current_cycle]) == 1:
                    cycles[current_cycle].add(random.choice(['A', 'C', 'G', 'T']))
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




                # def create_sets(cycles, s1, s2, s3):
#
#
#     strings = s1, s2, s3
#     n = max(len(s) for s in strings)
#     result = []
#     for i in range(n):
#         letters = cycles[i]
#         for s in strings:
#             if i < len(s):
#                 letters.add(s[i])
#         while len(letters) == 1:
#             letters.add(random.choice(['A', 'C', 'G', 'T']))
#         result.append(letters)






if __name__ == '__main__':
    s1 = "CCTAG"
    s2 = "ATAAT"
    s3 = "GAAGT"
    x = leading_indices(s1, s2, s3)
    # if x == {}:
    #     return separate_strings(s1, s2, s3)
    first_letters = [key[0] for key in x.keys()]
    l1 = int(first_letters[0])
    l2 = int(first_letters[1])
    lcs_indices = [value for value in x.values()]
    lcs1 = lcs_indices[0]
    lcs2 = lcs_indices[1]
    print(lcs1)
    print(lcs2)