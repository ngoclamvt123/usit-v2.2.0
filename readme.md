% University of Salzburg Iris Toolkit (USIT) Version 2.1.0


License
=======

H. Hofbauer, C. Rathgeb, A. Uhl, and P. Wild,
University of Salzburg, AUSTRIA, 
2016

Copyright (c) 2016, University of Salzburg
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

> Redistributions of source code must retain the above copyright notice, this
> list of conditions and the following disclaimer.
> Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation
> and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If this software is used to prepare for an article please include the following reference:

Text
----

C. Rathgeb, A. Uhl, P. Wild, and H. Hofbauer. "Design Decisions for an Iris Recognition SDK," in K. Bowyer and M. J. Burge, editors, Handbook of iris recognition, second edition, Advances in Computer Vision and Pattern Recognition, Springer, 2016.

Bibtex
------

    @incollection{USIT2,
        author     = {Christian Rathgeb and Andreas Uhl and Peter Wild and Heinz Hofbauer},
        title      = {Design Decisions for an Iris Recognition SDK},
        booktitle  = {Handbook of Iris Recognition},
        editor     = {Kevin Bowyer and Mark J. Burge},
        publisher  = {Springer},
        year       = {2016},
        series     = {Advances in Computer Vision and Pattern Recognition},
        edition    = {second edition},
    }


Requirements
============

These programs require the following libraries:

 - [OpenCV version 2.4.9](http://opencv.org)
 - [Boost version 1.59](http://www.boost.org)  
    specifically:
    - filesystem
    - system
    - regex


Algorithm description
=====================


Segmentation
------------

 * _caht_ ... Contrast-adjusted Hough Transform
 * _wahet_ .. Weighted Adaptive Hough and Ellipsopolar Transform
 * _ifpp_ ... Iterative Fourier-series Push Pull
 * _manuseg_ ... Uses points from a manual segmentation to extract the iris texture.


Iris Mask comparions
--------------------

 * _maskcmp_ ... Comparison of iris masks


Iris Feature Extracation
------------------------

 * _lg_ ... 1D-LogGabor Feature Extraction (=> hd for comparison)
 * _cg_ ... Complex Gabor filterbanks as used by Daugman (=> hd for comparison)
 * _qsw_ ... Extraction with the algorithm of Ma et al. (=> hd for comparison)
 * _ko_ ... Algorithm of ko et al. (=> koc for comparison)
 * _cr_ ... Algorithm of Rathgeb and Uhl (=> hd for comparison)
 * _cb_ ... Context-based Iris Recognition (=> cbc for comparison)
 * _dct_ ... Algorithm of Monroe et al. (=> dctc for comparison)
 * _sift_ ... Sift points as iris code (=> siftc for comparison)
 * _surf_ ... Surf points as iris code (=> surfc for comparison)
 * _lbp_ ... Local binary pattern based features (=> lbpcc for comparison)


Comparators
-----------

 * _koc_ ... Algorithm of Ko et al.
 * _cbc_ ... Context based algorithm
 * _dctc_ ... Algorithm of Monro et al.
 * _siftc_ ... Comparator for sift iris codes
 * _surfc_ ... Comparator for surf iris codes
 * _lbpc_ ... Comparator for lbp based iris codes
 * _hd_ ... Hamming Distance-based Comparator


Verification
------------

 * _hdverify_ ... Performance of Hamming Distance-based verification of iris codes


Face/Face-part detection
------------------------

 * _gfcf_ ... Gaussian Face and Face-part Classification Fusion


