#!/bin/bash

for th in 100;do
	
	for v in 1 20 50;do
		
		for f in `ls $1`;do	
			echo "!@#$ $1/$f, $th, $v"
			python FindBestMatch.py $th $v "$1/$f"
		done		
		
	done 

done
