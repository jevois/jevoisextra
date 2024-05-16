#!/bin/sh

set -e

# Get dlib if not here:
if [ ! -d dlib ]; then ./download-dlib.sh; fi

sudo /bin/rm -rf phbuild \
    && mkdir phbuild \
    && cd phbuild\
    && cmake "$@" -DJEVOIS_HARDWARE=PRO .. \
    && make -j \
    && sudo make install
