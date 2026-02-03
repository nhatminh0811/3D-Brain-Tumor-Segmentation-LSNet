import os
p='data/BraTS2020/TrainingData'
if not os.path.exists(p):
    print('TrainingData dir not found:', p)
else:
    subs=sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))])
    print('num subdirs:', len(subs))
    print('first5:', subs[:5])
    for d in subs[:5]:
        subp=os.path.join(p,d)
        try:
            print('\n--', d)
            print(os.listdir(subp)[:50])
        except Exception as e:
            print('error listing', e)
