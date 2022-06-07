import random

with open('tracks.txt') as f:
    contents = f.read().splitlines()
random.shuffle(contents)

with open('tracks_output.txt', 'w') as f:
    for track in contents:
        f.write(track)
        f.write('\n')