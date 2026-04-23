import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mido


LOGGER = logging.getLogger(__name__)


EXPERT_GUITAR_MIN = 96
EXPERT_GUITAR_MAX = 100
FORCE_HOPO_ON = 101
FORCE_HOPO_OFF = 102


@dataclass
class NoteEvent:
	time_tick: int
	time_sec: float
	lanes: Tuple[int, ...]
	force_hopo: Optional[bool]


def _find_part_guitar_track(mid: mido.MidiFile, track_name: str) -> Optional[mido.MidiTrack]:
	target = track_name.strip().lower()
	for track in mid.tracks:
		if track.name and track.name.strip().lower() == target:
			return track
	return None


def _collect_tempo_events(mid: mido.MidiFile) -> List[Tuple[int, int]]:
	tempo_events: List[Tuple[int, int]] = []
	for track in mid.tracks:
		abs_tick = 0
		for msg in track:
			abs_tick += msg.time
			if msg.type == "set_tempo":
				tempo_events.append((abs_tick, msg.tempo))

	if not tempo_events:
		raise ValueError("No tempo marker found in MIDI (set_tempo meta required).")

	tempo_events.sort(key=lambda x: x[0])
	if tempo_events[0][0] != 0:
		raise ValueError("First tempo marker must be at tick 0.")
	return tempo_events


def _ticks_to_seconds(
	ticks: List[int],
	tempo_events: List[Tuple[int, int]],
	ticks_per_beat: int,
) -> List[float]:
	if not ticks:
		return []

	ticks_sorted = sorted(ticks)
	seconds = [0.0] * len(ticks_sorted)

	tempo_index = 0
	last_tick = tempo_events[0][0]
	current_tempo = tempo_events[0][1]
	elapsed_sec = 0.0

	for idx, tick in enumerate(ticks_sorted):
		while tempo_index + 1 < len(tempo_events) and tick >= tempo_events[tempo_index + 1][0]:
			next_tick, next_tempo = tempo_events[tempo_index + 1]
			delta_ticks = next_tick - last_tick
			sec_per_tick = (current_tempo / 1_000_000.0) / ticks_per_beat
			elapsed_sec += delta_ticks * sec_per_tick
			last_tick = next_tick
			current_tempo = next_tempo
			tempo_index += 1

		delta_ticks = tick - last_tick
		sec_per_tick = (current_tempo / 1_000_000.0) / ticks_per_beat
		seconds[idx] = elapsed_sec + delta_ticks * sec_per_tick

	# Map back to original order
	tick_to_sec = dict(zip(ticks_sorted, seconds))
	return [tick_to_sec[tick] for tick in ticks]


def _build_note_events(
	mid: mido.MidiFile,
	track_name: str,
) -> Tuple[List[NoteEvent], int, int, int]:
	track = _find_part_guitar_track(mid, track_name)
	if track is None:
		raise ValueError(f"Track '{track_name}' not found in MIDI.")

	tempo_events = _collect_tempo_events(mid)
	ticks_per_beat = mid.ticks_per_beat

	note_events: Dict[int, set] = {}
	marker_starts: Dict[int, int] = {}
	marker_ranges: List[Tuple[int, int, bool]] = []

	abs_tick = 0
	force_on_count = 0
	force_off_count = 0

	for msg in track:
		abs_tick += msg.time
		if msg.is_meta:
			continue
		if msg.type not in ("note_on", "note_off"):
			continue
		note = msg.note
		is_note_on = msg.type == "note_on" and msg.velocity > 0
		is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)

		if is_note_on:
			if EXPERT_GUITAR_MIN <= note <= EXPERT_GUITAR_MAX:
				note_events.setdefault(abs_tick, set()).add(note - EXPERT_GUITAR_MIN)
			elif note == FORCE_HOPO_ON:
				marker_starts[note] = abs_tick
				force_on_count += 1
			elif note == FORCE_HOPO_OFF:
				marker_starts[note] = abs_tick
				force_off_count += 1
		elif is_note_off:
			if note in (FORCE_HOPO_ON, FORCE_HOPO_OFF) and note in marker_starts:
				start_tick = marker_starts.pop(note)
				marker_ranges.append((start_tick, abs_tick, note == FORCE_HOPO_ON))

	final_tick = abs_tick
	for note, start_tick in marker_starts.items():
		marker_ranges.append((start_tick, final_tick, note == FORCE_HOPO_ON))

	marker_ranges.sort(key=lambda item: item[0])

	if not note_events:
		return [], force_on_count, force_off_count, ticks_per_beat

	ticks = list(note_events.keys())
	seconds = _ticks_to_seconds(ticks, tempo_events, ticks_per_beat)
	tick_to_sec = dict(zip(ticks, seconds))

	def resolve_force_hopo(tick: int) -> Optional[bool]:
		forced_value: Optional[bool] = None
		for start_tick, end_tick, forced in marker_ranges:
			if start_tick <= tick <= end_tick:
				forced_value = forced
		return forced_value

	note_event_list: List[NoteEvent] = []
	for tick in sorted(note_events.keys()):
		lanes = tuple(sorted(note_events[tick]))
		force_hopo = resolve_force_hopo(tick)
		note_event_list.append(
			NoteEvent(
				time_tick=tick,
				time_sec=tick_to_sec[tick],
				lanes=lanes,
				force_hopo=force_hopo,
			)
		)

	return note_event_list, force_on_count, force_off_count, ticks_per_beat


def _compute_active_time(note_times: List[float], break_gap_sec: float) -> float:
	if len(note_times) < 2:
		return 0.0

	total_duration = note_times[-1] - note_times[0]
	if total_duration <= 0:
		return 0.0

	break_time = 0.0
	for prev, current in zip(note_times, note_times[1:]):
		gap = current - prev
		if gap >= break_gap_sec:
			break_time += gap

	return max(total_duration - break_time, 0.0)


def _rolling_peak_count(times: List[float], window_sec: float) -> float:
	if not times:
		return 0.0
	start = 0
	max_count = 0
	for end in range(len(times)):
		while times[end] - times[start] > window_sec:
			start += 1
		max_count = max(max_count, end - start + 1)
	return max_count / window_sec


def _rolling_peak_weighted(events: List[Tuple[float, float]], window_sec: float) -> float:
	if not events:
		return 0.0
	start = 0
	running_sum = 0.0
	max_sum = 0.0
	for end in range(len(events)):
		running_sum += events[end][1]
		while events[end][0] - events[start][0] > window_sec:
			running_sum -= events[start][1]
			start += 1
		max_sum = max(max_sum, running_sum)
	return max_sum / window_sec


def extract_midi_features(
	midi_path: str,
	hopo_threshold: int = 170,
	window_sec: float = 3.0,
	break_gap_sec: float = 5.0,
	track_name: str = "PART GUITAR",
	debug: bool = False,
) -> Dict[str, float]:
	mid = mido.MidiFile(midi_path)
	note_events, force_on_count, force_off_count, ticks_per_beat = _build_note_events(mid, track_name)

	if not note_events:
		features = {
			"total_active_time": 0.0,
			"avg_nps": 0.0,
			"peak_nps": 0.0,
			"avg_strums_per_sec": 0.0,
			"peak_strums_per_sec": 0.0,
			"avg_fret_changes_per_sec": 0.0,
			"peak_fret_changes_per_sec": 0.0,
		}
		if debug:
			features["debug"] = {
				"track_name": track_name,
				"chord_count": 0,
				"note_count": 0,
				"force_on_count": force_on_count,
				"force_off_count": force_off_count,
				"hopo_threshold": hopo_threshold,
				"ticks_per_beat": ticks_per_beat,
			}
		return features

	note_times: List[float] = []
	for note_event in note_events:
		note_times.extend([note_event.time_sec] * len(note_event.lanes))
	note_times.sort()

	active_time = _compute_active_time(note_times, break_gap_sec)
	total_notes = len(note_times)

	avg_nps = (total_notes / active_time) if active_time > 0 else 0.0
	peak_nps = _rolling_peak_count(note_times, window_sec)

	strum_event_times: List[float] = []
	hopo_note_count = 0
	change_events: List[Tuple[float, float]] = []

	previous_note_event: Optional[NoteEvent] = None
	for note_event in note_events:
		event_note_count = len(note_event.lanes)

		if previous_note_event is None:
			is_strum = True
		else:
			prev_size = len(previous_note_event.lanes)
			delta_ticks = note_event.time_tick - previous_note_event.time_tick
			is_single_to_single = event_note_count == 1 and prev_size == 1
			is_different_note = note_event.lanes != previous_note_event.lanes

			if note_event.force_hopo is True:
				is_strum = False
			elif note_event.force_hopo is False:
				is_strum = True
			elif not is_single_to_single:
				is_strum = True
			else:
				is_strum = not (is_different_note and delta_ticks <= hopo_threshold)

		if is_strum:
			strum_event_times.append(note_event.time_sec)
		else:
			hopo_note_count += event_note_count

		if previous_note_event is not None:
			previous_lanes = set(previous_note_event.lanes)
			current_lanes = set(note_event.lanes)
			# Count pitches that differ from the previous event.
			change_weight = len(current_lanes.symmetric_difference(previous_lanes))
			if change_weight > 0:
				change_events.append((note_event.time_sec, float(change_weight)))

		previous_note_event = note_event

	avg_strums_per_sec = (len(strum_event_times) / active_time) if active_time > 0 else 0.0
	peak_strums_per_sec = _rolling_peak_count(sorted(strum_event_times), window_sec)

	total_fret_changes = sum(weight for _, weight in change_events)
	avg_fret_changes_per_sec = (total_fret_changes / active_time) if active_time > 0 else 0.0
	peak_fret_changes_per_sec = _rolling_peak_weighted(change_events, window_sec)

	features = {
		"total_active_time": round(active_time, 4),
		"avg_nps": round(avg_nps, 6),
		"peak_nps": round(peak_nps, 6),
		"avg_strums_per_sec": round(avg_strums_per_sec, 6),
		"peak_strums_per_sec": round(peak_strums_per_sec, 6),
		"avg_fret_changes_per_sec": round(avg_fret_changes_per_sec, 6),
		"peak_fret_changes_per_sec": round(peak_fret_changes_per_sec, 6),
	}

	if debug:
		chord_event_count = sum(1 for note_event in note_events if len(note_event.lanes) > 1)
		single_note_event_count = len(note_events) - chord_event_count
		features["debug"] = {
			"track_name": track_name,
			"event_count": len(note_events),
			"chord_event_count": chord_event_count,
			"single_note_event_count": single_note_event_count,
			"note_count": total_notes,
			"active_time_sec": round(active_time, 4),
			"break_gap_sec": break_gap_sec,
			"window_sec": window_sec,
			"strum_event_count": len(strum_event_times),
			"strum_chord_event_count": sum(
				1
				for note_event in note_events
				if len(note_event.lanes) > 1 and note_event.time_sec in strum_event_times
			),
			"strum_single_note_event_count": sum(
				1
				for note_event in note_events
				if len(note_event.lanes) == 1 and note_event.time_sec in strum_event_times
			),
			"hopo_note_count": hopo_note_count,
			"total_fret_changes": round(total_fret_changes, 6),
			"force_on_count": force_on_count,
			"force_off_count": force_off_count,
			"hopo_threshold": hopo_threshold,
			"ticks_per_beat": ticks_per_beat,
			"total_notes": total_notes,
			"total_strum_events": len(strum_event_times),
			"total_strum_chord_events": sum(
				1
				for note_event in note_events
				if len(note_event.lanes) > 1 and note_event.time_sec in strum_event_times
			),
			"total_strum_single_note_events": sum(
				1
				for note_event in note_events
				if len(note_event.lanes) == 1 and note_event.time_sec in strum_event_times
			),
			"total_chord_events": chord_event_count,
			"total_single_note_events": single_note_event_count,
			"total_hopo_notes": hopo_note_count,
			"total_events": len(note_events),
			"sample_events": [
				{
					"time_sec": round(note_event.time_sec, 4),
					"lanes": note_event.lanes,
					"force_hopo": note_event.force_hopo,
				}
				for note_event in note_events[:10]
			],
		}

	return features


def _configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(levelname)s %(message)s",
	)


def _main() -> None:
	parser = argparse.ArgumentParser(description="Extract RB3 Expert PART GUITAR features from a MIDI file.")
	parser.add_argument("--midi", required=True, help="Path to the .mid file")
	parser.add_argument("--hopo", type=int, default=170, help="HOPO threshold in ticks (default: 170)")
	parser.add_argument("--window", type=float, default=3.0, help="Rolling window in seconds")                  #THIS IS X AND Y
	parser.add_argument("--break-gap", type=float, default=5.0, help="Break gap threshold in seconds")
	parser.add_argument("--track", default="PART GUITAR", help="Track name to parse")
	parser.add_argument("--debug", action="store_true", help="Print debug info")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
	args = parser.parse_args()

	_configure_logging(args.verbose)
	features = extract_midi_features(
		midi_path=args.midi,
		hopo_threshold=args.hopo,
		window_sec=args.window,
		break_gap_sec=args.break_gap,
		track_name=args.track,
		debug=args.debug,
	)

	if args.debug:
		debug_info = features.pop("debug", {})
		LOGGER.info("Features:\n%s", json.dumps(features, indent=2, sort_keys=True))
		LOGGER.info("Debug:\n%s", json.dumps(debug_info, indent=2, sort_keys=True))
	else:
		LOGGER.info("Features: %s", features)


if __name__ == "__main__":
	_main()
