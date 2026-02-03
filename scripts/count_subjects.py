import sys
sys.path.append('src')
from dataset import get_brats2020_datalist

if __name__ == '__main__':
    dl = get_brats2020_datalist('data/BraTS2020')
    print('valid subjects:', len(dl))
    if len(dl) < 10:
        print('Example entries:', dl[:3])
