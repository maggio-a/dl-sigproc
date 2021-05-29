# pconv-inpainter

1. Download dataset:
```
mkdir dataset
wget https://www.dropbox.com/s/5i1ciqhqksmdtmj/Places365_val_large.tar -P dataset
tar -xf dataset/Places365_val_large.tar -C dataset
```

2. Train
```
python pconv_main.py
```
or
```
python context_encoders_main.py
```

