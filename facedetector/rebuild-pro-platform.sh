#!/bin/sh
# USAGE: rebuild-pro-platform.sh [cmake opts]

# compiled code is in /var/lib/jevoispro-build/ and /var/lib/jevoispro-microsd/

set -e

# Get dlib if not here:
if [ ! -d dlib ]; then ./download-dlib.sh; fi

sudo /bin/rm -rf ppbuild \
    && mkdir ppbuild \
    && cd ppbuild \
    && cmake "$@" -DJEVOIS_HARDWARE=PRO -DJEVOIS_PLATFORM=ON .. \
    && make -j \
    && sudo make install
