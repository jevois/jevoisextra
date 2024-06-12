#!/bin/bash
# USAGE: rebuild-platform.sh [--jvpkg] [cmake opts]
#
# If --jvpkg is specified, then a jevois .jvpkg package will also be created

set -e

create_jvpkg="no"
if [ "X$1" = "X--jvpkg" ]; then create_jvpkg="yes"; shift; fi

# Get dlib if not here:
if [ ! -d dlib ]; then ./download-dlib.sh; fi

# On ARM hosts like Raspberry Pi3, we will likely run out of memory if attempting more than 1 compilation thread:
ncpu=`cat /proc/cpuinfo |grep processor|wc -l`
if [ `cat /proc/cpuinfo | grep ARM | wc -l` -gt 0 ]; then ncpu=1; fi

sudo /bin/rm -rf pbuild
mkdir pbuild
cd pbuild
cmake "${extra} $@" -DJEVOIS_PLATFORM=ON ..
make -j
sudo make install
cd ..

if [ $create_jvpkg = "yes" ]; then jevois-jvpkg `cat pbuild/jvpkg-args`; fi
