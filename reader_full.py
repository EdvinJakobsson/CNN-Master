import csv

def read_dataset(*args, filepath): #takes either two integers for first and last essay, or a list of integers for which prompts to include

    if len(args) == 2:
        if not isinstance(args[0], int):
            print("read_dataset: wrong input")
            exit()
        data = []
        start = args[0]
        end = args[1]
        counter = 0
        if start < 1:
            start = 1
        with open(filepath, newline='', encoding='latin1') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                counter += 1
                if counter <= start:
                    continue
                data.append(row)
                if counter == end+1:
                    break

    elif isinstance(args[0], list) and len(args) == 1:
        data = []
        skipfirstline = True
        with open(filepath, newline='', encoding='latin1') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if skipfirstline == True:
                    skipfirstline = False
                    continue
                if row[1] == 1:
                    print("tes")
                if int(row[1]) not in args[0]:
                    continue
                data.append(row)
    else:
            print("read_dataset: wrong input")


    return data
