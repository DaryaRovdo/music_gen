import mido
import numpy as np

from data_processing.constants import MAX_NOTE, MIN_NOTE, N, DEFAULT_TEMPO


def _get_new_state(msg: mido.Message, last_state: np.array):
    """
    Get new roll state based on previous state and new note event
    :param msg: note event
    :param last_state: array of shape (N), where N - length of range of supported notes
    :return: new state, array of shape (N), where N - length of range of supported notes
    """
    if 'note_on' == msg.type or 'note_off' == msg.type:
        new_state = last_state.copy()
        if MIN_NOTE <= msg.note <= MAX_NOTE:
            new_state[msg.note - MIN_NOTE] = msg.velocity if msg.type == 'note_on' else 0
        return new_state
    return last_state


def _check_quick_off_on(prev_msg: mido.Message, current_msg: mido.Message, roll: np.array):
    """
    Check if it was a quick note release and press again. If so, add tiny 1 tick pause between two presses.
    :param prev_msg: previous note event
    :param current_msg: current note event
    :param roll: current piano roll so far, array of shape (n_ticks, N),
     where n_ticks - num of ticks, N - length of range of supported notes
    """
    if prev_msg.type == 'note_off' and current_msg.type == 'note_on' and prev_msg.note == current_msg.note:
        roll[-1, prev_msg.note - MIN_NOTE] = 0


def _get_diff_states(states: np.array):
    """
    Get difference between each previous and next states
    :param states: states array of shape (n_ticks, N), where n_ticks - num of ticks, N - length of range of supported notes
    :return: array representing difference between each previous and next states, same shape as :param states array
    """
    new_ary = np.concatenate([np.array([[0] * N]), states], axis=0)
    return new_ary[1:] - new_ary[:-1]


def track_to_roll(track: mido.MidiTrack):
    """
    Convert midi track to piano roll
    :param track: midi track
    :return: np array of shape (n_ticks, N), where n_ticks - num of ticks, N - length of range of supported notes
    """
    roll = None
    last_state = _get_new_state(track[0], [0] * N)
    for i in range(1, len(track)):
        new_state = _get_new_state(track[i], last_state)
        if track[i].time > 0:
            add_result = np.tile(last_state, (track[i].time, 1))
            roll = np.vstack([roll, add_result]) if roll is not None else add_result
        _check_quick_off_on(track[i - 1], track[i], roll)
        last_state = new_state
    return roll


def roll_to_track(roll: np.array):
    """
    Convert piano roll with velocities to midi track
    :param roll: np array of shape (n_ticks, N), where ticks - num of ticks, N - length of range of supported notes
    :return: midi track
    """
    changes = _get_diff_states(roll)

    track = mido.MidiTrack()
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:
            last_time += 1
        else:
            notes = np.where(ch != 0)[0]
            notes_vol = ch[notes]
            first = True
            for n, v in zip(notes, notes_vol):
                new_time = last_time if first else 0
                track.append(mido.Message('note_on' if v > 0 else 'note_off', note=n + MIN_NOTE, velocity=max(v, 0), time=new_time))
                first = False
            last_time = 0
    return track


def track_to_midi(track: mido.MidiTrack, tempo: int = DEFAULT_TEMPO):
    """
    Save track to midi as guitar track
    :param track: track to be saved
    :param tempo: tempo of the midi
    :return: midi file
    """
    midi = mido.MidiFile()
    tempo_track = mido.MidiTrack()
    midi.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    tempo_track.append(mido.MetaMessage('track_name', name='Generated Guitar', time=0))
    tempo_track.append(mido.MetaMessage('instrument_name', name='Electric Guitar', time=0))
    midi.tracks.append(track)

    return midi


def midi_to_track(path: str):
    """
    Get important guitar track from midi
    :param path: path to midi file
    :return: guitar track
    """
    mid = mido.MidiFile(path, clip=True)
    best_track = None
    second_best_track = None
    third_best_track = None
    for track in mid.tracks:
        if 'guitar' in track.name.lower() and ('rhythm' in track.name.lower() or 'rythm' in track.name.lower()):
            best_track = track
        if 'guitar' in track.name.lower() and 'clean' in track.name.lower():
            second_best_track = track
        if 'guitar' in track.name.lower():
            third_best_track = track
    return best_track if best_track is not None else (second_best_track if second_best_track is not None else third_best_track)


if __name__ == '__main__':
    path = ''
    track = midi_to_track(path)
    roll = track_to_roll(track)
    new_track = roll_to_track(roll)
    new_midi = track_to_midi(track)
    new_midi.save('test.mid')

