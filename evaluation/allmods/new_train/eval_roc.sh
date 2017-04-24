for i in `ls *iter.h5`; do
        echo '###################################'
        echo $i
        for j in 0.2 0.5 0.7 0.9; do
		echo $j
		python ROCeval.py 70 20 $i 10000 $j
	done
        echo '###################################'
done
