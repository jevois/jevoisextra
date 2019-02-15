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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <atomic>

#include <box.h>
#include <pthread.h>
#include <additionally.h>

// Missing from additionally.h:
list *read_data_cfg(char *filename);
void free_network(network net);
char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);

// icon by yolo creators

static jevois::ParameterCategory const ParamCateg("Darknet YOLO Options");

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(dataroot, std::string, "Root path for data, config, and weight files. If empty, use "
                         "the module's path.",
                         JEVOIS_SHARE_PATH "/darknet/yololight", ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(datacfg, std::string, "Data configuration file (if relative, relative to dataroot)",
                         "cfg/coco.data", ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(cfgfile, std::string, "Network configuration file (if relative, relative to dataroot)",
                         "cfg/yolov3-tiny.cfg", ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(weightfile, std::string, "Network weights file (if relative, relative to dataroot)",
                         "weights/yolov3-tiny.weights", ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(namefile, std::string, "Category names file, or empty to fetch it from the network "
                         "config file (if relative, relative to dataroot)",
                         "", ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(nms, float, "Non-maximum suppression intersection-over-union threshold in percent",
                         20.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(thresh, float, "Detection threshold in percent confidence",
                         24.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(hierthresh, float, "Hierarchical detection threshold in percent confidence",
                         50.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(netin, cv::Size, "Width and height (in pixels) of the neural network input layer, or [0 0] "
                         "to make it match camera frame size. NOTE: for YOLO v3 sizes must be multiples of 32.",
                         cv::Size(320, 224), ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(quantized, bool, "Use INT8 quantized inference (faster but slightly less accurate)",
                         true, ParamCateg);

//! Parameter \relates YoloLight
JEVOIS_DECLARE_PARAMETER(letterbox, bool, "Letterbox the input frame instead of stretching it.",
                         false, ParamCateg);


//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! Darknet is a popular neural network framework, and YOLO is a very interesting network that detects all objects in a
    scene in one pass. This module detects all instances of any of the objects it knows about (determined by the
    network structure, labels, dataset used for training, and weights obtained) in the image that is given to it.

    See https://pjreddie.com/darknet/yolo/

    This module runs a YOLO network and shows all detections obtained. The YOLO network is currently quite slow, hence
    it is only run once in a while. Point your camera towards some interesting scene, keep it stable, and wait for YOLO
    to tell you what it found.  The framerate figures shown at the bottom left of the display reflect the speed at which
    each new video frame from the camera is processed, but in this module this just amounts to converting the image to
    RGB, sending it to the neural network for processing in a separate thread, and creating the demo display. Actual
    network inference speed (time taken to compute the predictions on one image) is shown at the bottom right. See
    below for how to trade-off speed and accuracy.

    Note that by default this module runs tiny-YOLO V3 which can detect and recognize 80 different kinds of objects from
    the Microsoft COCO dataset. This module can also run tiny-YOLO V2 for COCO, or tiny-YOLO V2 for the Pascal-VOC
    dataset with 20 object categories. See the module's \b params.cfg file to switch network.

    - The 80 COCO object categories are: person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic,
      fire, stop, parking, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
      handbag, tie, suitcase, frisbee, skis, snowboard, sports, kite, baseball, baseball, skateboard, surfboard, tennis,
      bottle, wine, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot, pizza, donut,
      cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell,
      microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy, hair, toothbrush.

    - The 20 Pascal-VOC object categories are: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
      diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

    Sometimes it will make mistakes! The performance of yolov3-tiny is about 33.1% correct (mean average precision) on
    the COCO test set.

    \youtube{d5CfljT5kec}

    Speed and network size
    ----------------------

    The parameter \p netin allows you to rescale the neural network to the specified size. Beware that this will only
    work if the network used is fully convolutional (as is the case of the default tiny-yolo network). This not only
    allows you to adjust processing speed (and, conversely, accuracy), but also to better match the network to the input
    images (e.g., the default size for tiny-yolo is 416x416, and, thus, passing it a input image of size 640x480 will
    result in first scaling that input to 416x312, then letterboxing it by adding gray borders on top and bottom so that
    the final input to the network is 416x416). This letterboxing can be completely avoided by just resizing the network
    to 320x240.

    Here are expected processing speeds for yolov2-tiny-voc:
    - when netin = [0 0], processes letterboxed 416x416 inputs, about 2450ms/image
    - when netin = [320 240], processes 320x240 inputs, about 1350ms/image
    - when netin = [160 120], processes 160x120 inputs, about 695ms/image

    YOLO V3 is faster, more accurate, uses less memory, and can detect 80 COCO categories:
    - when netin = [320 240], processes 320x240 inputs, about 870ms/image

    \youtube{77VRwFtIe8I}

    Serial messages
    ---------------

    When detections are found which are above threshold, one message will be sent for each detected
    object (i.e., for each box that gets drawn when USB outputs are used), using a standardized 2D message:
    + Serial message type: \b 2D
    + `id`: the category of the recognized object, followed by ':' and the confidence score in percent
    + `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
    + `w`, `h`: standardized object size
    + `extra`: any number of additional category:score pairs which had an above-threshold score for that box
    
    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    @author Laurent Itti

    @displayname Darknet YOLO
    @videomapping NONE 0 0 0.0 YUYV 640 480 0.4 JeVois YoloLight
    @videomapping YUYV 1280 480 15.0 YUYV 640 480 15.0 JeVois YoloLight
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2019 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class YoloLight : public jevois::StdModule,
                  jevois::Parameter<dataroot, datacfg, cfgfile, weightfile, namefile, nms, thresh,
                                    hierthresh, netin, quantized, letterbox>
{
  public: 
    // ####################################################################################################
    //! Constructor
    YoloLight(std::string const & instance) : jevois::StdModule(instance), names(nullptr), nboxes(0),
        dets(nullptr), classes(0), map(nullptr), itsReady(false)
    { }

    // ####################################################################################################
    //! Initialize, configure and load the network in a thread
    /*! Any call to process() will simply throw until the network is loaded and ready */
    void postInit() override
    {
      itsReadyFut = std::async(std::launch::async, [&]() {
          std::string root = dataroot::get(); if (root.empty() == false) root += '/';
          
          // Note: darknet expects read/write pointers to the file names...
          std::string const datacf = absolutePath(root + datacfg::get());
          std::string const cfgfil = absolutePath(root + cfgfile::get());
          std::string const weightfil = absolutePath(root + weightfile::get());
          
          list * options = read_data_cfg(const_cast<char *>(datacf.c_str()));
          std::string name_list = namefile::get();
          if (name_list.empty()) name_list = absolutePath(root + option_find_str(options, "names", "data/names.list"));
          else name_list = absolutePath(root + name_list);
          
          LINFO("Using data config from " << datacf);
          LINFO("Using cfg from " << cfgfil);
          LINFO("Using weights from " << weightfil);
          LINFO("Using names from " << name_list);
          
          LINFO("Getting labels...");
          names = get_labels(const_cast<char *>(name_list.c_str()));
          
          char * mapf = option_find_str(options, "map", 0);
          if (mapf) map = read_map(mapf);
          
          LINFO("Parsing network and loading weights...");
          int const quant = quantized::get() ? 1 : 0;
          net = parse_network_cfg(const_cast<char *>(cfgfil.c_str()), 1, quant);
          load_weights_upto_cpu(&net, const_cast<char *>(weightfil.c_str()), net.n);

          /*
          if (net == nullptr)
          {
            free_list(options);
            if (map) { free(map); map = nullptr; }
            LFATAL("Failed to load YOLO network and/or weights -- ABORT");
          }
          */
          classes = option_find_int(options, "classes", 2);
          
          //set_batch_network(net, 1);
          srand(2222222);
          yolov2_fuse_conv_batchnorm(net);
          calculate_binary_weights(net);
          if (quant) { LINFO("Quantization!"); quantinization_and_get_multipliers(net); }
          LINFO("YOLO network ready");
          free_list(options);
          itsReady.store(true);
        });
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    virtual ~YoloLight()
    { }

    // ####################################################################################################
    //! Un-initialize and free resources
    void postUninit() override
    {
      if (itsReadyFut.valid()) try { itsReadyFut.get(); } catch (...) { }
      try { itsPredictFut.get(); } catch (...) { }
      if (dets) { free_detections(dets, nboxes); dets = nullptr; nboxes = 0; }
      if (map) { free(map); map = nullptr; }
      free_network(net);
      free_ptrs((void**)names, classes);
      names = nullptr; classes = 0;
    }
    
    // ####################################################################################################
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects an OpenCV RGB byte image which will be converted to float RGB planar, and which may be
        letterboxed if necessary to fit network input dims. Returns the prediction time (neural net forward pass) in
        milliseconds. Throws std::logic_error if the network is still loading and not ready. */
    float predict(cv::Mat const & cvimg)
    {
      if (itsReady.load() == false) throw std::logic_error("not ready yet...");
      if (cvimg.type() != CV_8UC3) LFATAL("cvimg must have type CV_8UC3 and RGB pixels");
      
      int const c = 3; // color channels
      int const w = cvimg.cols;
      int const h = cvimg.rows;
      
      image im = make_image(w, h, c);
      for (int k = 0; k < c; ++k)
        for (int j = 0; j < h; ++j)
          for (int i = 0; i < w; ++i)
          {
            int dst_index = i + w*j + w*h*k;
            int src_index = k + c*i + c*w*j;
            im.data[dst_index] = float(cvimg.data[src_index]) / 255.0F;
          }
      
      float predtime = predict(im);
      free_image(im);
      return predtime;
    }
    
    // ####################################################################################################
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects a Darknet image input, RGB float planar normalized to [0..1], with same dims as network
        input dims. Returns the prediction time (neural net forward pass) in milliseconds. Throws std::logic_error if
        the network is still loading and not ready. */
    float predict(image & im)
    {
      image sized; bool need_free = false;
      if (im.w == net.w && im.h == net.h) sized = im;
      else { sized = resize_image(im, net.w, net.h); need_free = true; } // used to be letterbox_image()
      
      struct timeval start, stop;
      
      gettimeofday(&start, 0);
      if (quantized::get()) network_predict_quantized(net, sized.data);
      else network_predict_cpu(net, sized.data);
      gettimeofday(&stop, 0);
      
      float predtime = (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
      
      if (need_free) free_image(sized);
      
      return predtime;
    }
    
    // ####################################################################################################
    //! Compute the boxes
    /*! You must have called predict() first for this to not violently crash. */
    void computeBoxes(int inw, int inh)
    {
      layer & l = net.layers[net.n-1];
      if (dets) { free_detections(dets, nboxes); dets = nullptr; nboxes = 0; }
      int const letter = letterbox::get() ? 1 : 0;
      dets = get_network_boxes(&net, inw /* was 1*/, inh /* was 1*/, thresh::get() * 0.01F, hierthresh::get() * 0.01F, map, 0, &nboxes, letter);
      float const nmsval = nms::get() * 0.01F;
      if (nmsval) do_nms_sort(dets, nboxes, l.classes, nmsval);
    }

    // ####################################################################################################
    //! Draw the detections
    /*! You must have called computeBoxes() first for this to not violently crash. */
    void drawDetections(jevois::RawImage & outimg, int inw, int inh, int xoff, int yoff)
    {
      float const thval = thresh::get();

      for (int i = 0; i < nboxes; ++i)
      {
        // For each detection, We need to get a list of labels and probabilities, sorted by score:
        std::vector<jevois::ObjReco> data;
        
        for (int j = 0; j < classes; ++j)
        {
          float const p = dets[i].prob[j] * 100.0F;
          if (p > thval) data.push_back( { p, std::string(names[j]) } );
        }
        
        // End here if nothing above threshold:
        if (data.empty()) continue;
        
        // Sort in descending order:
        std::sort(data.begin(), data.end(), [](auto a, auto b) { return (b.score < a.score); });
        
        // Create our display label:
        std::string labelstr;
        for (auto const & d : data)
        {
          if (labelstr.empty() == false) labelstr += ", ";
          labelstr += jevois::sformat("%s:%.1f", d.category.c_str(), d.score);
        }
        
        box const & b = dets[i].bbox;
        
        int const left = std::max(xoff, int(xoff + (b.x - b.w / 2.0F) * inw + 0.499F));
        int const bw = std::min(inw, int(b.w * inw + 0.499F));
        int const top = std::max(yoff, int(yoff + (b.y - b.h / 2.0F) * inh + 0.499F));
        int const bh = std::min(inh, int(b.h * inh + 0.499F));
        
        jevois::rawimage::drawRect(outimg, left, top, bw, bh, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::writeText(outimg, labelstr,
                                    left + 6, top + 2, jevois::yuyv::LightGreen, jevois::rawimage::Font10x20);
      }
    }
    
    // ####################################################################################################
    //! Send serial messages about detections
    /*! You must have called computeBoxes() first for this to not violently crash. The module given should be the owner
        of this component, we will use it to actually send each serial message using some variant of
        jevois::Module::sendSerial(). */
    void sendSerial(jevois::StdModule * mod, int inw, int inh)
    {
      float const thval = thresh::get();
      
      for (int i = 0; i < nboxes; ++i)
      {
        // For each detection, We need to get a list of labels and probabilities, sorted by score:
        std::vector<jevois::ObjReco> data;
        
        for (int j = 0; j < classes; ++j)
        {
          float const p = dets[i].prob[j] * 100.0F;
          if (p > thval) data.push_back({ p, std::string(names[j]) });
        }
        
        // End here if nothing above threshold:
        if (data.empty()) continue;
        
        // Sort in descending order:
        std::sort(data.begin(), data.end(), [](auto a, auto b) { return (b.score < a.score); });
        
        box const & b = dets[i].bbox;
        
        int const left = (b.x - b.w / 2.0F) * inw;
        int const bw = b.w * inw;
        int const top = (b.y - b.h / 2.0F) * inh;
        int const bh = b.h * inh;
        
        mod->sendSerialObjDetImg2D(inw, inh, left, top, bw, bh, data);
      }
    }
    
    // ####################################################################################################
    //! Resize the network's input image dims
    /*! This will prepare the network to receive inputs of the specified size. It is optional and will be called
        automatically by predict() if the given image size does not match the current network input size. Note that this
        only works with fully convolutional networks. Note that the number of channels cannot be changed at this
        time. Throws std::logic_error if the network is still loading and not ready. */
    void resizeInDims(int w, int h)
    {
      if (itsReady.load() == false) throw std::logic_error("not ready yet...");
      /////// not supported??? resize_network(net, w, h);
    }
    
    // ####################################################################################################
    //! Get input width, height, channels
    /*! Throws std::logic_error if the network is still loading and not ready. */
    void getInDims(int & w, int & h, int & c) const
    {
      if (itsReady.load() == false) throw std::logic_error("not ready yet...");
      w = net.w; h = net.h; c = net.c;
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      int ready = true; float ptime = 0.0F;

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Convert input image to RGB for predictions:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

      // Resize the network and/or the input if desired:
      cv::Size nsz = netin::get();
      if (nsz.width != 0 && nsz.height != 0)
      {
        resizeInDims(nsz.width, nsz.height);
        itsNetInput = jevois::rescaleCv(cvimg, nsz);
      }
      else
      {
        resizeInDims(cvimg.cols, cvimg.rows);
        itsNetInput = cvimg;
      }

      cvimg.release();
      
      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions, will throw logic_error if we are still loading the network:
      try { ptime =  predict(itsNetInput); } catch (std::logic_error const & e) { ready = false; }

      if (ready)
      {
        LINFO("Predicted in " << ptime << "ms");

        // Compute the boxes:
        computeBoxes(w, h);

        // Send serial results:
        sendSerial(this, w, h);
      }
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 50, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only handle one specific pixel format, and any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w * 2, h, inimg.fmt);

          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - input", 3, 3, jevois::yuyv::White);

          // Paste the latest prediction results, if any, otherwise a wait message:
          cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
          if (itsRawPrevOutputCv.empty() == false)
            itsRawPrevOutputCv.copyTo(outimgcv(cv::Rect(w, 0, w, h)));
          else
          {
            jevois::rawimage::drawFilledRect(outimg, w, 0, w, h, jevois::yuyv::Black);
            jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - loading network - please wait...",
                                        w + 3, 3, jevois::yuyv::White);
          }
        });

      // Decide on what to do based on itsPredictFut: if it is valid, we are still predicting, so check whether we are
      // done and if so draw the results. Otherwise, start predicting using the current input frame:
      if (itsPredictFut.valid())
      {
        // Are we finished predicting?
        if (itsPredictFut.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready)
        {
          // Do a get() on our future to free up the async thread and get any exception it might have thrown. In
          // particular, it will throw a logic_error if we are still loading the network:
          bool success = true; float ptime = 0.0F;
          try { ptime = itsPredictFut.get(); } catch (std::logic_error const & e) { success = false; }

          // Wait for paste to finish up:
          paste_fut.get();
          
          // Let camera know we are done processing the input image:
          inframe.done();
          
          if (success)
          {
            cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
            
            // Update our output image: First paste the image we have been making predictions on:
            if (itsRawPrevOutputCv.empty()) itsRawPrevOutputCv = cv::Mat(h, w, CV_8UC2);
            itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, w, h)));
            
            // Then draw the detections:
            drawDetections(outimg, w, h, w, 0);
            
            // Send serial messages:
            sendSerial(this, w, h);
            
            // Draw some text messages:
            jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - predictions", w + 3, 3, jevois::yuyv::White);
            jevois::rawimage::writeText(outimg, "YOLO predict time: " + std::to_string(int(ptime)) + "ms",
                                        w + 3, h - 13, jevois::yuyv::White);
            
            // Finally make a copy of these new results so we can display them again while we wait for the next round:
            outimgcv(cv::Rect(w, 0, w, h)).copyTo(itsRawPrevOutputCv);
          }
        }
        else
        {
          // Future is not ready, do nothing except drawings on this frame (done in paste_fut thread) and we will try
          // again on the next one...
          paste_fut.get();
          inframe.done();
        }
      }
      else
      {
        // Note: resizeInDims() could throw if the network is not ready yet.
        try
        {
          // Convert input image to RGB for predictions:
          cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);
          
          // Also make a raw YUYV copy of the input image for later displays:
          cv::Mat inimgcv = jevois::rawimage::cvImage(inimg);
          inimgcv.copyTo(itsRawInputCv);
          
          // Resize the network if desired:
          cv::Size nsz = netin::get();
          if (nsz.width != 0 && nsz.height != 0)
          {
            resizeInDims(nsz.width, nsz.height);
            itsNetInput = jevois::rescaleCv(cvimg, nsz);
          }
          else
          {
            resizeInDims(cvimg.cols, cvimg.rows);
            itsNetInput = cvimg;
          }
          
          cvimg.release();
          
          // Launch the predictions:
          itsPredictFut = std::async(std::launch::async, [&](int ww, int hh)
                                     {
                                       float pt = predict(itsNetInput);
                                       computeBoxes(ww, hh);
                                       return pt;
                                     }, w, h);
        }
        catch (std::logic_error const & e) { }
        
        // Wait for paste to finish up:
        paste_fut.get();
        
        // Let camera know we are done processing the input image:
        inframe.done();
      }
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  protected:
    network net;
    char ** names;
    int nboxes;
    detection * dets;
    int classes;
    int * map;
    std::future<void> itsReadyFut;
    std::atomic<bool> itsReady;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsRawPrevOutputCv;
    cv::Mat itsNetInput;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(YoloLight);
