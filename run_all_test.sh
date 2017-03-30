#!/bin/sh

result_file="test_result.txt"

echo "--------All testcases result------\n" > $result_file

for files in ~/TTTN/Iris-Recog/USITv2.2.0/Test/*.tiff
do
	ncfile=$(basename $files)
	echo "Matching result:\t"$ncfile"\n" >> $result_file
	## Matching ##
	sh matching.sh $files
	##############
	echo "------------------------------------\n" >> $result_file
done