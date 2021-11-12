import os
import sys
import numpy as np
import librosa
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import patches

import libfmp.b
import libfmp.c3
import libfmp.c7

# constants
Fs = 22050
## chroma feature coefficients
N_feat = 4410
H_feat = 2205

## CENS coefficients
ell = 21 # smoothing
d = 5 # downsample

def load_audio(audio_name, data_dir="songs"):
    fn = os.path.join(data_dir, audio_name)
    sig, _ = librosa.load(fn, sr=Fs)
    return sig

def DTW_cost_matrix(x1, x2, Fs, ell=21, d=5):
    # x1 & x2: audio signals
    # ell: smoothing size
    # d: downsample rate
    C1 = librosa.feature.chroma_stft(y=x1, sr=Fs, tuning=0, norm=None, hop_length=H_feat, n_fft=N_feat) # 12 chroma bins * time frames with frame size = hopsize / Fs
    C2 = librosa.feature.chroma_stft(y=x2, sr=Fs, tuning=0, norm=None, hop_length=H_feat, n_fft=N_feat)
    
    X, Fs_cens = libfmp.c7.compute_cens_from_chromagram(C1, ell=ell, d=d)
    Y, Fs_cens = libfmp.c7.compute_cens_from_chromagram(C2, ell=ell, d=d)
    
    C_FMP = libfmp.c3.compute_cost_matrix(X, Y, 'euclidean')
    return C_FMP

def min_distance_loc(C):
    return np.where(C==np.min(C))[0][0], np.where(C==np.min(C))[1][0]

def index_to_time(idx):
    return idx*d*H_feat / Fs

def index_to_sample(idx):
    return idx*d*H_feat

def time_to_idx(t):
    return t * Fs / d / H_feat

def plot_min_distance(C, t_x1, t_x2, offset=60):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12, 6))
    plt.imshow(C, cmap='gray_r', origin='lower', aspect='equal')
    plt.plot(t_x1, t_x2, marker='o', color='r', alpha=0.7)
    plt.clim([0, np.max(C)])
    plt.colorbar()

    # annotation 
    bbox = dict(boxstyle ="round", fc ="0.8",color="r")
    arrowprops = dict(
        # arrowstyle = "->",
        facecolor ='w',
        width=1,
        edgecolor='w',
        shrink=0.1
    )

    plt.annotate(f'({t_x1}, {t_x2})', (t_x1, t_x2), xytext =(t_x1+offset, t_x2+offset),
                bbox=bbox,
                arrowprops=arrowprops)

    plt.title('Cost matrix $C$ of the two chroma sequences')
    plt.xlabel('Song B')
    plt.ylabel('Song A')

def segment(sig, start=0.0, duration=10.0):
    start = int(start*Fs) if start > 0.0 else 0
    duration = int(duration*Fs)
    return sig[start: start+duration]

def DTW_matching(x1, x2, ell=21, d=5):
    # ell: smoothing size
    # d: downsample rate
    C1 = librosa.feature.chroma_stft(y=x1, sr=Fs, tuning=0, norm=None, hop_length=H_feat, n_fft=N_feat) # 12 chroma bins * time frames with frame size = hopsize / Fs
    C2 = librosa.feature.chroma_stft(y=x2, sr=Fs, tuning=0, norm=None, hop_length=H_feat, n_fft=N_feat)
    
    X, Fs_cens = libfmp.c7.compute_cens_from_chromagram(C1, ell=ell, d=d)
    Y, Fs_cens = libfmp.c7.compute_cens_from_chromagram(C2, ell=ell, d=d)
    N, M = X.shape[1], Y.shape[1]
    
    C_FMP = libfmp.c3.compute_cost_matrix(X, Y, 'euclidean')
    sigma = np.array([[1, 0], [0, 1], [1, 1]])
    D_librosa, P_librosa = librosa.sequence.dtw(C=C_FMP, step_sizes_sigma=sigma, subseq=True, backtrack=True)
    P_librosa = P_librosa[::-1, :]
    
    return D_librosa, P_librosa

def plot_DTW_path(D, P):
    # D: aacumulative cost matrix
    # P: warping path
    plt.rcParams.update({'font.size': 16})
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    libfmp.c3.plot_matrix_with_points(D, P, ax=[ax], cmap=cmap, 
                                      xlabel=f'Song A', ylabel=f'Song B', 
                                      # title=title,
                                      marker='o', linestyle='-')
    ax.set_title("DTW Optimal Warping Path")

def fragment_match(x1, x2, t, fragment_duration=20, ell=21, d=5):
    fragment = segment(x2, t-fragment_duration/2, fragment_duration) # t from preprocessing
    D, P = DTW_matching(fragment, x1, ell, d)
    return fragment, D, P

def DTW_subsequence(D, P):
    # D: aacumulative cost matrix
    # P: warping path
    N = D.shape[0]
    Delta_DTW = D[-1, :] / N # matching function
    a_ast = P[0, 1]
    b_ast = P[-1, 1]
    return Delta_DTW, a_ast, b_ast

def plot_DTW_subsequence(Delta_DTW, a_ast, b_ast):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(15, 5))
    libfmp.b.plot_signal(Delta_DTW, ax=ax, xlabel='Time (samples)', ylabel='', ylim=[0, 5],
                         title=r'Matching function $\Delta_\mathrm{DTW}$', color='k')
    ax.set_xlim([-0.5, len(Delta_DTW)-0.5])
    ax.grid()
    ax.plot(b_ast, Delta_DTW[b_ast], 'ro')
    ax.add_patch(patches.Rectangle((a_ast-0.5, 0), b_ast-a_ast+1, 7, facecolor='r', alpha=0.2))
    # annotation style
    bbox = dict(boxstyle ="round", fc ="0.8")
    arrowprops = dict(
        arrowstyle = "->",
        facecolor ='red',
        # connectionstyle = "angle, angleA = 0, angleB = 90,rad = 10"
    )

    offset = 60
    ax.annotate('Minimum', xy=(b_ast, Delta_DTW[b_ast]), xytext=(b_ast+15, 1.5),
            arrowprops=dict(facecolor='black', width=3, shrink=0.05))
    ax.text(10, -0.8, 'Song A')
    ax.text(a_ast-5, -0.5, r'$a^*$')
    ax.text(b_ast-5, -0.5, r'$b^*$')