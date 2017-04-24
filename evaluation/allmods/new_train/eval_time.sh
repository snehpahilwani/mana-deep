for i in `ls *iter.h5`; do
        echo '###################################'
        echo $i
        for j in `seq 1 10`; do
		echo $j
		python FindBestMatch.py 70 1 $i $j
		python FindBestMatch.py 70 20 $i $j
	done
        echo '###################################'
done
