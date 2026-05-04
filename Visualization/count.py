with open('11000L_1.csv', 'r') as f:
    data = f.read()
    values = data.split(',')
    print(len(values))