import csv

with open('/mnt/users/code/test/volumes.csv', 'r') as f:
    reader = csv.reader(f)  
    counts = []
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10 = 0,0,0,0,0,0,0,0,0,0,0
    for row in reader:
        if int(row[1]) <= 10:
            i0 += 1
        if int(row[1]) <= 20 and int(row[1]) > 10:
            i1 += 1
        if int(row[1]) <= 30 and int(row[1]) > 20:
            i2 += 1
        if int(row[1]) <= 40 and int(row[1]) > 30:
            i3 += 1
        if int(row[1]) <= 50 and int(row[1]) > 40:
            i4 += 1
        if int(row[1]) <= 100 and int(row[1]) > 50:
            i5 += 1
        if int(row[1]) <= 300 and int(row[1]) > 100:
            i6 += 1
        if int(row[1]) <= 500 and int(row[1]) > 300:
            i7 += 1
        if int(row[1]) <= 700 and int(row[1]) > 500:
            i8 += 1
        if int(row[1]) <= 1000 and int(row[1]) > 700:
            i9 += 1
        if int(row[1]) > 1000:
            i10 += 1  
    print(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10)

with open('/mnt/users/code/test/brightness.csv', 'r') as f:
    reader = csv.reader(f)  
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10 = 0,0,0,0,0,0,0,0,0,0,0
    for row in reader:
        if float(row[1]) <= 100:
            i0 += 1
        if float(row[1]) <= 200 and float(row[1]) > 100:
            i1 += 1
        if float(row[1]) <= 300 and float(row[1]) > 200:
            i2 += 1
        if float(row[1]) <= 400 and float(row[1]) > 300:
            i3 += 1
        if float(row[1]) <= 500 and float(row[1]) > 400:
            i4 += 1
        if float(row[1]) <= 600 and float(row[1]) > 500:
            i5 += 1
        if float(row[1]) <= 700 and float(row[1]) > 600:
            i6 += 1
        if float(row[1]) <= 800 and float(row[1]) > 700:
            i7 += 1
        if float(row[1]) <= 900 and float(row[1]) > 800:
            i8 += 1
        if float(row[1]) <= 1000 and float(row[1]) > 900:
            i9 += 1
        if float(row[1]) > 1000:
            i10 += 1  
    print(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10)