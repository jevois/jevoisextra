#!/bin/sh

# Get dlib if not here:
if [ ! -d dlib ]; then ./download-dlib.sh; fi

/bin/rm -rf hbuild \
    && mkdir hbuild \
    && cd hbuild \
    && cmake "$@" .. \
    && make -j \
    && sudo make install
