#%%
import pretty_midi
import numpy as np
import librosa.display


#%%
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    """ Plot piano roll from .mid file
    ----------
    Parameters:
        pm: RWC, MDB, iKala, DSD100
        start/end_pitch: lowest/highest note (float)
        fs: sampling freq. (int)
    
    """
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(
        pm.get_piano_roll(fs)[start_pitch:end_pitch],
        hop_length=1,
        sr=fs,
        x_axis="time",
        y_axis="cqt_note",
        fmin=pretty_midi.note_number_to_hz(start_pitch),
    )


def midi_to_note(file_name, pitch_shift, fs=100, start_note=40, end_note=95):
    """ Convert .mid to note
    ----------
    Parameters:
        file_name: '.mid' (str)
        pitch_sifht: shift the pitch to adjust notes correctly (int)
        fs: sampling freq. (int)
        start/end_pitch: lowest/highest note(int)
    
    ----------
    Returns: 
        notes: note/10ms (array)
    """

    pm = pretty_midi.PrettyMIDI(file_name)
    frame_note = pm.get_piano_roll(fs)[start_note:end_note]

    length_audio = frame_note.shape[1]
    notes = np.zeros(length_audio)

    for i in range(length_audio):
        note_tmp = np.argmax(frame_note[:, i])
        if note_tmp > 0:
            notes[i] = (note_tmp + start_note) + pitch_shift
            # note[i] = 2 ** ((note_tmp -69) / 12.) * 440
    return notes


def midi_to_segment(filename):
    """ Convert .mid to segment
    ----------
    Parameters:
        filename: .mid (str)
    
    ----------
    Returns: 
        segments: [start(s),end(s),pitch] (list) 
    """

    pm = pretty_midi.PrettyMIDI(filename)
    segment = []
    for note in pm.instruments[0].notes:
        segment.append([note.start, note.end, note.pitch])
    return segment


def segment_to_midi(segments, path_output, tempo=120):
    """ Convert segment to .mid
    ----------
    Parameters:
        segments: [start(s),end(s),pitch] (list) 
        path_output: path of save file (str)
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=int(tempo))
    inst_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    inst = pretty_midi.Instrument(program=inst_program)
    for segment in segments:
        note = pretty_midi.Note(
            velocity=100, start=segment[0], end=segment[1], pitch=np.int(segment[2])
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(f"{path_output}")


def note_to_segment(note):
    """ Convert note to segment
    ----------
    Parameters:
        note: note/10ms (array)
    ----------
    Returns: 
        segments: [start(s),end(s),pitch] (list) 
    """
    startSeg = []
    endSeg = []
    notes = []
    flag = -1

    if note[0] > 0:
        startSeg.append(0)
        notes.append(np.int(note[0]))
        flag *= -1
    for i in range(0, len(note) - 1):
        if note[i] != note[i + 1]:
            if flag < 0:
                startSeg.append(0.01 * (i + 1))
                notes.append(np.int(note[i + 1]))
                flag *= -1
            else:
                if note[i + 1] == 0:
                    endSeg.append(0.01 * i)
                    flag *= -1
                else:
                    endSeg.append(0.01 * i)
                    startSeg.append(0.01 * (i + 1))
                    notes.append(np.int(note[i + 1]))

    return list(zip(startSeg, endSeg, notes))


def note2Midi(frame_level_pitchscroe, path_output, tempo):
    # note = np.loadtxt(path_input_note)
    # note = note[:, 1]
    segment = note_to_segment(frame_level_pitchscroe)
    segment_to_midi(segment, path_output=path_output, tempo=tempo)


# def note2Midi(path_input_note, path_output, tempo):
#     note = np.loadtxt(path_input_note)
#     note = note[:, 1]
#     segment = note_to_segment(note)
#     segment_to_midi(segment, path_output=path_output, tempo=tempo)

