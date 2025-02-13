import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pretty_midi
import glob
import os

# MIDI 파일이 있는 디렉터리 경로
midi_dir = r"D:\maestro-v3.0.0"  # 여기에 MIDI 파일이 있는 폴더 경로 입력

# 'name_1.mid', 'name_2.mid' 순서대로 정렬된 파일 목록 가져오기
midi_files = sorted(glob.glob(os.path.join(midi_dir, "2006","*.midi"),recursive=True))

# MIDI 파일들을 순서대로 읽기
midi_data_list = []
for midi_file in midi_files:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        midi_data_list.append(midi_data)
        print(f"Loaded: {midi_file}")
    except Exception as e:
        print(f"Failed to load {midi_file}: {e}")