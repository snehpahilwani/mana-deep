for i in `ls model4.h5`; do
        echo '###################################'
        echo $i
        python FindBestMatch.py 100 1 $i
        python FindBestMatch.py 100 40 $i
        echo '###################################'
done
