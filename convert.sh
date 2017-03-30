#########################################################################
# File Name: convert.sh
#########################################################################
#!/bin/bash

database=$1

cd $database
for files in *
do
	ncfile=$(basename $files)
	suffix=".*"
	pfile=${ncfile%$suffix}".tiff"
	convert $ncfile -type TrueColor $pfile
done
mv *.tiff ~/TTTN/Iris-Recog/thousand_tiff/
