import { useCallback, useRef, useState } from "react";
import { useMotionValue } from "framer-motion";

// A real hands-free "call": tap once to start, then this auto-detects when you stop
// talking (simple energy-based VAD on the raw mic stream) and submits each utterance
// on its own - no manual tap per turn. Mirrors the Gradio UI's server-side Silero VAD
// end-pointing (see Supertonic-100m/voice_pipeline/vad.py), just done client-side here
// since the React dashboard talks to the backend over discrete HTTP requests instead
// of a persistent streaming connection.
//
// Deliberately not the Web Speech API: on Android Chrome that hands recognition off to
// the OS's registered speech service (usually the Google app), which visibly launches
// Google Assistant instead of staying in this page. This hook only ever uses
// getUserMedia + MediaRecorder + Web Audio, so it behaves identically everywhere.

// Trailing quiet required to end an utterance.
const SILENCE_MS = 1400;
const MAX_UTTERANCE_MS = 20000; // safety cutoff if someone just keeps talking
const CHECK_INTERVAL_MS = 50;
const SPEECH_RMS_THRESHOLD = 0.02;
const SPEECH_CONFIRM_CHECKS = 2; // filters brief pops/clicks from counting as speech starting
// Utterances shorter than this are noise/false-starts, not real speech - discard them
// client-side instead of round-tripping a doomed empty-transcript request. Cutting
// SILENCE_MS too low without this (an earlier mistake) meant ordinary micro-pauses
// mid-sentence chopped real speech into garbage fragments, transcribed to nothing, and
// silently looped "Thinking" -> "Listening" with no reply ever appearing.
const MIN_SPEECH_MS = 300;

// Barge-in: talking over the assistant while it's speaking should cut it off, the way
// interrupting a person does. This is deliberately harder to trigger than ordinary
// speech detection above, because a false positive here is much worse - it would chop
// off the assistant's reply for a cough or a door closing.
//
// An earlier attempt at this fired instantly and constantly: getUserMedia ran without
// echoCancellation, so the mic picked the assistant's own voice back up out of the
// speakers and the agent interrupted itself within ~100ms of starting every reply.
// Echo cancellation (enabled in startCall) removes that feedback path; the higher
// threshold and longer confirmation window below absorb whatever residue leaks past it
// on speakerphone-ish setups.
const BARGE_IN_RMS_THRESHOLD = 0.05; // vs 0.02 to merely start an utterance
const BARGE_IN_CONFIRM_CHECKS = 6; // ~300ms sustained, vs ~100ms
// Playback and the tail of the user's own speech overlap for a moment at the handover,
// so ignore the first stretch of every reply rather than letting the user interrupt
// themselves.
const BARGE_IN_GRACE_MS = 500;

interface UseVoiceCallOptions {
  onUtterance: (blob: Blob) => void;
  // Fired when the user talks over the assistant. The caller decides what that means
  // (stop playback, start listening again) - this hook only detects it.
  onInterrupt?: () => void;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export function useVoiceCall({ onUtterance, onInterrupt }: UseVoiceCallOptions) {
  const supported =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined" &&
    typeof AudioContext !== "undefined";

  const [inCall, setInCall] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  // Live mic amplitude, already computed here for the VAD. Published as a MotionValue so
  // the UI can show the user's own voice moving the orb without a re-render per sample.
  const micLevel = useMotionValue(0);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const intervalRef = useRef<number | null>(null);
  const utteranceStartRef = useRef<number>(0);
  const speechStartAtRef = useRef<number | null>(null);
  const lastLoudAtRef = useRef<number | null>(null);
  const loudStreakRef = useRef(0);
  const discardRef = useRef(false);
  // MediaRecorder.stop() finishes asynchronously, so the final onstop fires *after*
  // endCall() has returned and the UI has already settled to idle. Without this flag
  // that last partial segment was submitted as a normal utterance: status flipped to
  // "thinking", and the reply was then dropped by the caller's own hung-up guard, so
  // nothing ever moved it off "thinking" again. Hanging up must discard that tail.
  const endedRef = useRef(false);
  const busyRef = useRef(false); // paused while a reply is being fetched/played, so we don't hear ourselves
  const bargeInStreakRef = useRef(0);
  const busySinceRef = useRef(0);
  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;
  const onInterruptRef = useRef(onInterrupt);
  onInterruptRef.current = onInterrupt;

  const stopSegmentRecorder = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
    recorderRef.current = null;
  }, []);

  const startSegmentRecorder = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) return;
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
      chunksRef.current = [];
      if (endedRef.current) {
        // Call is over - drop whatever was mid-recording and don't restart a segment.
        discardRef.current = false;
        return;
      }
      if (discardRef.current) {
        // false start / noise blip, too short to be real speech - drop it and keep
        // listening in a fresh segment instead of round-tripping a doomed request
        discardRef.current = false;
        startSegmentRecorder();
        return;
      }
      if (blob.size > 0) onUtteranceRef.current(blob);
    };
    recorder.start();
    recorderRef.current = recorder;
    utteranceStartRef.current = performance.now();
    speechStartAtRef.current = null;
    lastLoudAtRef.current = null;
    loudStreakRef.current = 0;
  }, []);

  // Call this once the assistant's reply has finished playing (or errored out) to
  // resume listening for the next turn. Exposed so the caller can gate it on TTS
  // playback finishing, not just on the network call finishing.
  const resumeListening = useCallback(() => {
    busyRef.current = false;
    bargeInStreakRef.current = 0;
    if (streamRef.current) startSegmentRecorder();
  }, [startSegmentRecorder]);

  const pauseForReply = useCallback(() => {
    busyRef.current = true;
    busySinceRef.current = performance.now();
    bargeInStreakRef.current = 0;
    stopSegmentRecorder();
  }, [stopSegmentRecorder]);

  const startCall = useCallback(async () => {
    if (!supported || inCall) return;
    setPermissionError(null);
    try {
      // Bare `{ audio: true }` leaves the browser's own DSP off, so room noise, fan
      // hum and speaker bleed all went straight into the recording (and into the VAD's
      // RMS reading, which is what made it trigger on nothing). These are handled in
      // native code by the browser before we ever see samples - free, and far better
      // than anything we'd do to the PCM ourselves.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          noiseSuppression: true,
          echoCancellation: true, // also stops the assistant's own playback feeding back in
          autoGainControl: true, // evens out quiet/far-from-mic speech before STT sees it
        },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      busyRef.current = false;
      endedRef.current = false;
      startSegmentRecorder();
      setInCall(true);

      const buf = new Uint8Array(analyser.fftSize);
      intervalRef.current = window.setInterval(() => {
        analyser.getByteTimeDomainData(buf);
        let sumSquares = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sumSquares += v * v;
        }
        const rms = Math.sqrt(sumSquares / buf.length);
        micLevel.set(rms);
        const now = performance.now();

        if (busyRef.current) {
          // No segment recorder runs while the assistant holds the floor, but the
          // analyser is wired to the raw stream rather than the recorder, so it can
          // still watch for the user cutting in.
          if (now - busySinceRef.current < BARGE_IN_GRACE_MS) return;
          if (rms > BARGE_IN_RMS_THRESHOLD) {
            bargeInStreakRef.current += 1;
            if (bargeInStreakRef.current >= BARGE_IN_CONFIRM_CHECKS) {
              bargeInStreakRef.current = 0;
              onInterruptRef.current?.();
            }
          } else {
            bargeInStreakRef.current = 0;
          }
          return;
        }
        if (!recorderRef.current) return;
        const loud = rms > SPEECH_RMS_THRESHOLD;

        if (loud) {
          loudStreakRef.current += 1;
          if (loudStreakRef.current >= SPEECH_CONFIRM_CHECKS) {
            if (speechStartAtRef.current === null) speechStartAtRef.current = now;
            lastLoudAtRef.current = now;
          }
        } else {
          loudStreakRef.current = 0;
        }

        const elapsed = now - utteranceStartRef.current;
        const trailingSilence = lastLoudAtRef.current ? now - lastLoudAtRef.current : 0;
        const hadSpeech = lastLoudAtRef.current !== null;

        if (hadSpeech && trailingSilence >= SILENCE_MS) {
          const speechDuration = lastLoudAtRef.current! - (speechStartAtRef.current ?? now);
          if (speechDuration < MIN_SPEECH_MS) {
            discardRef.current = true;
            stopSegmentRecorder();
          } else {
            pauseForReply();
          }
        } else if (elapsed >= MAX_UTTERANCE_MS) {
          pauseForReply();
        }
      }, CHECK_INTERVAL_MS);
    } catch (err) {
      micLevel.set(0);
      setPermissionError(
        err instanceof Error ? err.message : "Microphone access was denied or unavailable."
      );
      setInCall(false);
    }
  }, [supported, inCall, startSegmentRecorder, pauseForReply, micLevel]);

  const endCall = useCallback(() => {
    endedRef.current = true;
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stopSegmentRecorder();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
    busyRef.current = false;
    micLevel.set(0);
    setInCall(false);
  }, [stopSegmentRecorder, micLevel]);

  return {
    supported,
    inCall,
    permissionError,
    micLevel,
    startCall,
    endCall,
    pauseForReply,
    resumeListening,
  };
}
