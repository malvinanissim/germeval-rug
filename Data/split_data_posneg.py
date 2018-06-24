offense, other = [],[]
with open('germeval.ensemble.test.txt', 'r', encoding='utf-8') as fi:
    for line in fi:
        dat = line.strip().split('\t')
        if dat[1] == 'OTHER':
            other.append(dat[0])
        elif dat[1] == 'OFFENSE':
            offense.append(dat[0])
        else:
            raise ValueError('Unknown label!')

print(len(offense))
print(len(other))


print('Offense')
for item in offense[:10]:
    print(item)

print()

print('Other')
for item in other[:10]:
    print(item)

with open('offense.test.txt', 'w', encoding='utf-8') as fo1:
    for item in offense:
        print(item, file=fo1)

with open('other.test.txt', 'w', encoding='utf-8') as fo2:
    for item in other:
        print(item, file=fo2)
