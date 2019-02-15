#!/bin/bash
# usage: reinstall.sh [-y]
# will nuke and re-install all contributed packages

# Bump this release number each time you make significant changes here, this will cause rebuild-host.sh to re-run
# this reinstall script:
release=`cat RELEASE`

###################################################################################################
function get_github # owner, repo, revision
{
    echo "### JeVois: downloading ${1} / ${2} ..."
    git clone --recursive "https://github.com/${1}/${2}.git"
    if [ "X${3}" != "X" ]; then
        echo "### JeVois: moving ${1} / ${2} to checkout ${3} ..."
        cd "${2}"
        git checkout -q ${3}
        cd ..
    fi
}

###################################################################################################
function patchit # directory
{
    if [ ! -d ${1} ]; then
	    echo "Ooops cannot patch ${1} because directory is missing";
    else
        echo "### JeVois: patching ${1} ..."
	    cd ${1}
	    patch -p1 < ../${1}.patch
	    cd ..
    fi
}

###################################################################################################
if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to nuke, fetch and patch contributed packages [y/N]? "
fi


if [ "X$REPLY" = "Xy" ]; then
    ###################################################################################################
    # Cleanup:
    /bin/rm -rf libdmtx pylibdmtx yolo2_light

    ###################################################################################################
    # Get the packages:

    # Core library to decode DataMatrix tags
    get_github dmtx libdmtx 0c8cb2f542e74ee49ea5b2290a9c60d69b74fc01

    # Python bindings to libdmtx
    get_github NaturalHistoryMuseum pylibdmtx 6b1bb59fc7b0c55c56bc09eb6ded173144149555

    # YOLO2-light with XNOR support
    get_github AlexeyAB yolo2_light 85fc1b388aa00f0243220ed48dac95ac18401b22

    ###################################################################################################
    # Patching:
    for f in *.patch ; do
	    patchit ${f/.patch/}
    done
    
    ###################################################################################################
    # Keep track of the last installed release:
    echo $release > .installed
fi


