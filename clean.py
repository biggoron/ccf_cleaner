import os
from itertools import chain, combinations
import numpy as np
from matplotlib import pyplot as plt
from pyannote.core import Segment, notebook
import torch
import scipy.io.wavfile
from scipy.spatial.distance import cdist
import IPython.display as ipd

def get_models():
    dia = torch.hub.load('pyannote/pyannote-audio', 'dia_ami')
    emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
    return dia, emb

def clean_folder(input_folder, output_folder, dia, emb):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    for f in input_files:
        print(f)
        clean_file(
            os.path.join(input_folder, f),
            os.path.join(output_folder, f.replace('.wav', '_cleaned.wav')),
            dia, emb
            )
    
def clean_file(input_file, output_file, dia, emb):
    fs1, y1 = scipy.io.wavfile.read(input_file)
    diarization = dia({'audio': input_file})
    labels = diarization.labels()
    tracks = {
        label: [
            seg
            for seg in diarization.itersegments()
            if list(diarization.get_labels(seg))[0] == label]
        for label in labels
    }
    embeddings = emb({'audio': input_file})
    track_embeddings = [get_embedding(embeddings, tracks[i]) for i in labels]
    token = True
    while token:
        labels, tracks, track_embeddings, token = reduce_tracks(labels, tracks, embeddings, track_embeddings)
    label, speech_segments = max(tracks.items(), key=lambda item: sum([seg.duration for seg in item[1]]))
    speech_segments = sorted(speech_segments, key=lambda seg: seg.start)
    start_token = True
    buffer_output_segs = []
    for s in speech_segments:
        emb = embeddings.crop(s, mode='strict')
        if emb.shape[0] == 0:
            if start_token:
                continue
            else:
                buffer_output_segs += [s]
        else:
            if dist(np.mean(emb, axis=0), track_embeddings[labels.index(label)]) < 0.4:
                buffer_output_segs += [s]
    scipy.io.wavfile.write(
        output_file,
        fs1,
        y1[list(chain(*[
            range(int(seg.start * fs1), int(seg.end * fs1))
            for seg in buffer_output_segs
        ]))])

def get_embedding(embeddings, track):
    loc_embeddings = [embeddings.crop(seg, mode='strict') for seg in track]
    loc_embeddings = np.concatenate(loc_embeddings, axis=0)
    return np.mean(loc_embeddings, axis=0)

def dist(a, b):
    return cdist(np.expand_dims(a, axis=0), np.expand_dims(b, axis=0), metric='cosine')

def reduce_tracks(labels, tracks, embeddings, track_embeddings):
    for a, b in list(combinations(range(len(labels)), 2)):
        if dist(track_embeddings[a], track_embeddings[b]) < 0.08:
            tracks[labels[a]] += tracks[labels[b]]
            tracks.pop(labels[b])
            labels.pop(b)
            track_embeddings.pop(b)
            track_embeddings[a] = get_embedding(embeddings, tracks[labels[a]])
            return labels, tracks, track_embeddings, True
    return labels, tracks, track_embeddings, False
