for i in `ls *.h5`; do
        echo '###################################'
        echo $i
        python FindBestMatch.py 100 1 $i
        python FindBestMatch.py 100 20 $i
        echo '###################################'
done
