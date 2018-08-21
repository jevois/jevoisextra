#!/bin/bash
# Download and extract dlib:

ver=19.15

if [ -d dlib ]; then
    echo "dlib/ directory already present! Delete it if you want to reinstall -- SKIPPING DLIB DOWNLOAD"
    exit 0
fi

wget http://dlib.net/files/dlib-${ver}.tar.bz2
tar jxvf dlib-${ver}.tar.bz2
/bin/rm dlib-${ver}.tar.bz2
mv dlib-${ver} dlib
