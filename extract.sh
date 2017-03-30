#!/bin/sh

#Convert and move pictures from dataset to tiff folder

sh convert.sh "/home/sinhdn/TTTN/test-data-thousand"

suffix=".*"

START=$(date +%s.%N)

################## Segmentation ##################
##################################################

mkdir -p Texture
mkdir -p Mask

#Start repeat
for files in ~/TTTN/Iris-Recog/thousand_tiff/*.tiff
do
	ncfile=$(basename $files)

	## Extract texture and mask ##
	./caht -i $files -o ${ncfile%$suffix}_texture.tiff -m ${ncfile%$suffix}_mask.png -s 512 64 -e
	##############################

	mv ${ncfile%$suffix}_texture.tiff Texture
	mv ${ncfile%$suffix}_mask.png Mask
done
#End repeat

############# Feature Extraction #################
##################################################

mkdir -p IrisCode
mkdir -p MaskCode

#Start repeat
for files_t in ~/TTTN/Iris-Recog/USITv2.2.0/Texture/*.tiff
do
	ncfile_t=$(basename $files_t)
	icfile=${ncfile_t%$suffix}
	icfile=${icfile%????????}

	files_m=$(find ~/TTTN/Iris-Recog/USITv2.2.0/Mask -name $icfile"*")
	
	## Extract feature to iris code and mask code ##
	./lg -i $files_t -o $icfile"_code.png" -m $files_m  $icfile"_mskcode.png"
	##################################
	
	mv $icfile"_code.png" IrisCode
	mv $icfile"_mskcode.png" MaskCode
done
#End repeat

# mkdir -p MaskCode

# #Start repeat
# for files_m in ~/TTTN/Iris-Recog/USITv2.2.0/Mask/*.png
# do
# 	ncfile_m=$(basename $files_m)
# 	mcfile=${ncfile_m%$suffix}
# 	mcfile=${mcfile%?????}

# 	## Extract mask to mask code ##
# 	./lg -i $files_m -o $mcfile"_mskcode.png"
# 	##################################
	
# 	mv $mcfile"_mskcode.png" MaskCode
# done
# #End repeat

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

echo "It took $DIFF"