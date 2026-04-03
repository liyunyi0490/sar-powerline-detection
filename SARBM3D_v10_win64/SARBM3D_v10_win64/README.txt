SAR-BM3D
Date released 31/07/2013, version 1.0.
Functions for Matlab for the denoising of a SAR image corrupted
by multiplicative speckle noise with the technique described in 
"A Nonlocal SAR Image Denoising Algorithm Based on LLMMSE Wavelet Shrinkage", 
written by S. Parrilli, M. Poderico, C.V. Angelino and L. Verdoliva, 
IEEE Trans. on Geoscience and Remote Sensing, vol. 50, no. 2, pp. 606-616, 2012.
Please refer to this paper for a more detailed description of the algorithm.

Differences with the previous version:
- fixed bugs;
- optimization of the execution time;
- filtered output image at the first step is now available; 
- possibility to set the parameters of the algorithm both in the first and in the second step.

-------------------------------------------------------------------
 Copyright
-------------------------------------------------------------------

Copyright (c) 2012 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) and online at
http://www.grip.unina.it/download/LICENSE_CLOSED.txt

-------------------------------------------------------------------
 Installation
-------------------------------------------------------------------

Unzip the archive and add the folder to the search path of MATLAB.

-------------------------------------------------------------------
 Contents
-------------------------------------------------------------------

The package comprises the function "SARBM3D_v10".
For help on how to use this script, you can e.g. use "help SARBM3D_v10".

-------------------------------------------------------------------
 Requirements
-------------------------------------------------------------------

All the functions and scripts were tested on MATLAB 2011b,
the operation is not guaranteed with older version of MATLAB.
The software uses the OpenCV library (http://opencv.org/), all
necessary files are included in the archive.

The version for Windows 64 bit requires Microsoft
Visual C++ 2010 Redistributable Package (x64). It can be
downloaded from: 
	http://www.microsoft.com/download/details.aspx?id=14632
	
The version for Windows 32 bit requires Microsoft
Visual C++ 2010 Redistributable Package (x86). It can be
downloaded from:
	http://www.microsoft.com/download/details.aspx?id=5555

-------------------------------------------------------------------
 Execution times
-------------------------------------------------------------------

Execution times were evaluated on two different computers:
Machine 1: Intel(R) Core(TM) 2 Q9550 2,83Ghz 32bit; 3Gb Memory
Machine 2: Intel(R) Core(TM) i7-2600 3,40Ghz 64bit; 8Gb Memory

dim.    |  Mac. 1 |  Mac. 2 |
256x256 |  50 sec |  25 sec |
512x512 | 200 sec | 100 sec |

-------------------------------------------------------------------
 Feedback
-------------------------------------------------------------------

If you have any comment, suggestion, or question, please do
contact Luisa Verdoliva at verdoliv@unina.it
For other information you can see http://www.grip.unina.it/
