k = int(input())
choice = 0
for knum in range(10000,30001):
    sub1 = int(str(knum)[:3])
    sub2 = int(str(knum)[1:4])
    sub3 = int(str(knum)[2:5])
    if sub1%k ==0 and sub2%k ==0 and sub3 %k ==0:
        print(knum)
        choice = choice +1
if choice == 0:
    print('No')