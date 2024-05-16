######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2018 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, BA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################

# CMake build rules for libdmtx library

# You may provide the installed JeVois config root and version as:
#   cmake -DJEVOIS_CONFIG=/jevois/config ..

cmake_minimum_required(VERSION 3.6)

# Set vendor name, our modules will be placed in a directory by that name under /jevois/modules:
set(JEVOIS_VENDOR "JeVois")
if (JEVOIS_HARDWARE STREQUAL "PRO")
  set(JEVOIS_CONFIG "/jevoispro/config" CACHE STRING "Path to JeVois config to use")
else()
  set(JEVOIS_CONFIG "/jevois/config" CACHE STRING "Path to JeVois config to use")
endif()

# Include our helper functions, config, etc from the JeVois install:
set(CMAKE_MODULE_PATH ${JEVOIS_CONFIG})
include(jevois_config)
include(JeVois)

# Set project name, detects compiler (which has been set by our helper module). Then set some complation flags:
project(jvdmtx C)
jevois_project_set_flags()

# Setup our library using just dmtx.c, which includes all other source files, and call it libjvdmtx with
# a version number:

# Note: jevois_setup_library2() expects that only one source file needs tobe compiled. You can add moe later
# using target_sources() as is done in the CMakeLists.txt of jevoisbase. If you want to compile all the C/C++
# files in a directory into the library, use jevois_setup_library() as is done in jevoisbase.
jevois_setup_library2("${CMAKE_CURRENT_SOURCE_DIR}/../Contrib/libdmtx/dmtx.c" jvdmtx 1.0.0)

# Add includes to our install / distribution package:
#add_subdirectory(include)

# libdmtx needs libm:
target_link_libraries(jvdmtx m)

# Let the .c files of libdmtx find dmtx.h:
include_directories(../Contrib/libdmtx)

# We could optionally add some definitions here:
#add_definitions(-DDARKNET_NNPACK)

########################################################################################################################
# Documentation files for our install / distribution package
set(DOC_FILES README INSTALL COPYING)
if (JEVOIS_PLATFORM)
  set(DOC_PATH "share/doc/libjvdmtx-platform")
else (JEVOIS_PLATFORM)
  set(DOC_PATH "share/doc/libjvdmtx-host")
endif (JEVOIS_PLATFORM)
install(FILES ${DOC_FILES} DESTINATION ${DOC_PATH} COMPONENT bin)

########################################################################################################################
# Debian packaging:

# Create packages (Debian, RPM): in hbuild/ or pbuild/, just type 'sudo cpack' to create the package.
# To list the files created in a package, run: dpkg --contents <package.deb>
set(CPACK_PACKAGE_DESCRIPTION "JeVois Smart Machine Vision libjvdmtx")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "JeVois Smart Embedded Machine Vision Toolkit, libjvdmtx")
set(CPACK_PACKAGE_CONTACT "Laurent Itti <jevois.org@gmail.com>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
set(CPACK_DEBIAN_PACKAGE_SECTION "universe")
set(CPACK_PACKAGE_VENDOR "iLab at the University of Southern California")
set(CPACK_PACKAGE_VERSION_MAJOR "${JEVOIS_VERSION_MAJOR}")  # Note: libjvdmtx version tracks jevois version
set(CPACK_PACKAGE_VERSION_MINOR "${JEVOIS_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${JEVOIS_VERSION_PATCH}")
set(JEVOIS_PACKAGE_RELEASE "1") # packager revision number

if (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "jevois-platform (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
else (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "jevois-host (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
endif (JEVOIS_PLATFORM)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${JEVOIS_DEPEND}")

# Use helper from JeVois.cmake for all other settings:
jevois_setup_cpack("jevois-libjvdmtx")


########################################################################################################################
# Wrapping up:
jevois_project_finalize()  # call this last in your CMakeLists.txt, will un-mount the live microSD if applicable
