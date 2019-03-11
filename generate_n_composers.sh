#!/usr/bin/env bash

python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/debussy/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Claude\ DebussyMIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--3.midi --return_states --state_file=data/generations/debussy/dump

python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/bach/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Johann\ Sebastian\ BachMIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--1.midi --return_states --state_file=data/generations/bach/dump

python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/chopin/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Frédéric\ ChopinMIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--2.midi --return_states --state_file=data/generations/chopin/dump

python  magenta/models/performance_rnn/performance_rnn_generate.py --bundle_file=~/Downloads/bundle.mag --output_dir=data/generations/beethoven/ --config=performance_with_dynamics --num_outputs=$1 --num_steps=3000 --primer_midi=data/test/Ludwig\ van\ BeethovenMIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--2.midi --return_states --state_file=data/generations/beethoven/dump
