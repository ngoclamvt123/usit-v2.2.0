* [**v2.2.0**] 2016.01.12
    - Fixed the bug where an empty point list for ellipse fitting would cause manuseg to break. Now the one failing input is skipped and the rest runs through.
    - HD has now the capability to report the bitshift at which the optimal HD was found.
    - Fixed a bug where the _lg_, _cg_, _cr_ and _qsw_ features wrote the bitsequence for iris codes bytewise out of order, i.e. byte order was correct, however bit order per byte was wrong. This lead to alignment errors with HD rotation correction with the `-s` option.
    - Included a package for Binarized Statistical Image Features (`bsif` and `bsifc`) from the paper:
      
      > **Christian Rathgeb, Florian Struck, Christoph Busch**. “_Efficient BSIF-based Near-Infrared Iris Recognition_”, in Proceedings of International Conference on Image Processing Theory, Tools and Applications (IPTA'16), 2016.

* [**v2.1.0**] 2016.03.22
    - Included package for TripleA from the paper:

      > **C. Rathgeb, H. Hofbauer, A. Uhl, and C. Busch**. “_TripleA: Accelerated Accuracy-preserving Alignment for Iris-Codes_”, Proceedings of the 9th IAPR/IEEE International Conference on Biometrics (ICB'16), 2016.


* [**v2.0.0**] 2016.02.04
    - Scaling options added as used in the paper:

      > **Heinz Hofbauer**, **Fernando Alonso-Fernandez**, **Josef Bigun**,  and **Andreas Uhl**. "_Experimental Analysis Regarding the Influence of Iris Segmentation on the Recognition Rate_," in IET Biometrics, 2016.
      
    - Variable iris texture height support added as used in the paper:

      >  **Rudolf Schraml**, **Heinz Hofbauer**, **Alexander Petutschnigg**, and **Andreas Uhl**. "_Tree Log Identification Based on Digital Cross-Section Images of Log Ends Using Fingerprint and Iris Recognition Methods_," In Proceedings of the 16th International Conference on Computer Analysis of Images and Patterns (CAIP'15), pp. 752-765, LNCS, Springer Verlag, 2015

    - New tools:
        - _cg_
        - _lbp_ and _lbpc_
        - _surf_ and _surfc_
        - _sift_ and _siftc_
        - _manuseg_
    - Renamed _iffp_ to _ifpp_ (for iterative fourier push pull).
