// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>

#include <dlib/opencv/cv_image.h>
#include <dlib/array2d/array2d_generic_image.h>
#include <dlib/image_processing/frontal_face_detector.h>

// icon by dlib team

//! Face Detection using the Dlib C++ machine vision library
/*! Detect faces using the HOG (histogram of gradients) algorithm provided by dlib.

    This module is heavily inspired from
    http://dlib.net/face_detection_ex.cpp.html

    It is an example of how to use dlib with JeVois.

    For more information about dlib, see http://dlib.net - Dlib provides a large number of high-quality C++11 algorithms
    for machine vision, deep neural networks, image processing, and machine learning.

    @author Laurent Itti

    @videomapping YUYV 320 264 15.0 YUYV 320 240 15.0 JeVois FaceDetector
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class FaceDetector : public jevois::StdModule
{
  private:
    dlib::frontal_face_detector itsDetector;
    
  public:
    //! Constructor
    FaceDetector(std::string const & instance) : jevois::StdModule(instance)
    {
      itsDetector = dlib::get_frontal_face_detector();
    }
    
    //! Virtual destructor for safe inheritance
    virtual ~FaceDetector() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only support YUYV pixels in this example, any resolution:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", inimg.width, inimg.height, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", outimg.width, h + 24, V4L2_PIX_FMT_YUYV);

          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois dlib Face Detector", 3, 3, jevois::yuyv::White);

          // Blank out the bottom of the frame:
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height - h, jevois::yuyv::Black);
        });

      // Convert to OpenCV RGB:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

      // Reinterpret as dlib image:
      dlib::array2d<dlib::rgb_pixel> dlibimg;
      dlib::assign_image(dlibimg, dlib::cv_image<dlib::rgb_pixel>(cvimg));

      // Detect faces:
      std::vector<dlib::rectangle> dets = itsDetector(dlibimg);
      
      // Wait for paste thread to complete and let camera know we are done processing the input image:
      paste_fut.get(); inframe.done();

      // Draw the rectangles and send serial messages:
      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(dets.size()) + " faces",
                                  3, h+6, jevois::yuyv::White);

      for (auto const & r : dets)
      {
        jevois::rawimage::drawRect(outimg, r.left(), r.top(), r.width(), r.height(), 2, jevois::yuyv::LightGreen);
        sendSerialImg2D(w, h, r.left() - r.width() / 2, r.top() - r.height() / 2, r.width(), r.height(), "face");
      }
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(FaceDetector);
