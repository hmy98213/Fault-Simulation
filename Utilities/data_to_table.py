def data_to_table(file): 
    with open(file) as f:
        for line in f:
            l = line.rstrip().split('\t')
            # print(l)
            print("(%s, %s)"%(l[0], l[1]))

if __name__ == "__main__":
    data_to_table('table_data.txt')