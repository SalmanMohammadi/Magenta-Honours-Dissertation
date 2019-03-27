#!/usr/bin/env bash

#find $1 -type d -exec find -name *.midi -exec echo {} \;

#1 - 

for dir in $1*
do
	#dir=$("$dir"/*"")
	curdir="$(basename $dir"/")"
	mkdir $2$curdir
	for file in $dir"/*"
	do
		for f in $file
		do
			#echo $f
			python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=$3 --output_dir=$2$curdir"/" --config=performance_with_dynamics --num_outputs=1 --num_steps=1000 --primer_midi=$f --return_states --state_file=$2$curdir/dump #--eval --eval_split=20
		done
		##curdir="$(basename $dir"/")"
		#echo $file
		#python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=$2$curdir"/" --config=performance_with_dynamics --num_outputs=1 --num_steps=3000 --primer_midi=$file --return_states--state_file=$2$curdir/dump
	done
done

#python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/debussy/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Claude\ DebussyMIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--3.midi --return_states --state_file=data/generations/debussy/dump

#python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/bach/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Johann\ Sebastian\ BachMIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--1.midi --return_states --state_file=data/generations/bach/dump

#python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/chopin/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Frédéric\ ChopinMIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--2.midi --return_states --state_file=data/generations/chopin/dump

#python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/beethoven/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Ludwig\ van\ BeethovenMIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--2.midi --return_states --state_file=data/generations/beethoven/dump


