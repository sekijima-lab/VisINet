#!/bin/bash

cd patches/

wget -O resnet_utils.py https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/resnet_utils.py

wget -O resnet_v2.py https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/resnet_v2.py

for i in $(ls *.patch); do 
    echo "../${i%.*}.py"
    patch -u -f -o patched.py < $i
    mv patched.py "../${i%.*}.py"
done
