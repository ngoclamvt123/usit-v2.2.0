#!/bin/sh

input_file=$1
suffix=".*"

START=$(date +%s.%N)

ncinput_file=$(basename $input_file)
txinput_file=${ncinput_file%$suffix}_texture.tiff
mkinput_file=${ncinput_file%$suffix}_mask.png
icinput_file=${ncinput_file%$suffix}_code.png
mcinput_file=${ncinput_file%$suffix}_mskcode.png

################## Segmentation ##################
##################################################

./caht -i $input_file -o $txinput_file -m $mkinput_file -s 512 64 -e

############# Feature Extraction #################
##################################################

./lg -i $txinput_file -o $icinput_file -m $mkinput_file  $mcinput_file

################## Comparison ####################
##################################################

result_file=${ncinput_file%$suffix}_matching-result.txt

for files in ~/TTTN/Iris-Recog/USITv2.2.0/IrisCode/*.png
do
	ncfiles=$(basename $files)
	ncfiles=${ncfiles%$suffix}
	ncfiles=${ncfiles%?????}

	mkfiles=$(find ~/TTTN/Iris-Recog/USITv2.2.0/MaskCode -name $ncfiles"*")
	
	./hd -i $icinput_file $files -m $mcinput_file $mkfiles -s -3 3 >> $result_file
	# -m $mcinput_file $mkfiles 
done

group=${ncinput_file%$suffix}
group=${group%???}

./findBestMatching $result_file $group >> "test_result.txt"

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "It took $DIFF"

mv $txinput_file ${input_file%/*}
mv $mkinput_file ${input_file%/*}
mv $icinput_file ${input_file%/*}
mv $mcinput_file ${input_file%/*}
mv $result_file ${input_file%/*}