for i in `ls *iter.h5`; do
        echo '###################################'
        echo $i
        python FindBestMatch.py 70 1 $i
	python FindBestMatch.py 70 60 $i
	python FindBestMatch.py 100 50 $i
        echo '###################################'
done
