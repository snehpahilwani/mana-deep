for i in `ls *iter.h5`; do
        echo '###################################'
        echo $i
        #for j in `seq 1 10`; do
		#echo $j
		python FindBestMatch.py 70 1 $i 10000
		python FindBestMatch.py 70 20 $i 10000
	#done
        echo '###################################'
done
