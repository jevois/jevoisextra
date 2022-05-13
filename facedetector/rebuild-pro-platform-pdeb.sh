#!/bin/sh
# USAGE: rebuild-pro-platform-pdeb.sh [cmake opts]

# compiled code is in /var/lib/jevoispro-build-pdeb/

set -e

# Get dlib if not here:
if [ ! -d dlib ]; then ./download-dlib.sh; fi

sudo /bin/rm -rf ppdbuild \
    && mkdir ppdbuild \
    && cd ppdbuild \
    && cmake "$@" -DJEVOISPRO_PLATFORM_DEB=ON -DJEVOIS_HARDWARE=PRO -DJEVOIS_PLATFORM=ON .. \
    && make -j \
    && sudo make install \
    && sudo cpack

