rm op/*
for i in `ls`; do
#echo $i
convert -threshold 85% -extent 559x559 -gravity center $i -trim -fuzz 20% +fuzz xc:white -composite -resize 224x224  op/op_$i.tif
convert -threshold 85% -extent 244x244 -gravity center xc:white op/op_$i.tif -composite op/op_$i.tif
done 
