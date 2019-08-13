#Used when there is not enough space for labels on the x-axis
def multiliner(string_list, n):
    length = len(string_list)
    for i in range(length):
        rem = i % n
        string_list[i] = '\n' * rem + string_list[i]
    return string_list
