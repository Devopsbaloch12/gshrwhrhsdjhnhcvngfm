import { useCallback, useRef, useState } from "react";

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
const SILENCE_MS = 350;
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

interface UseVoiceCallOptions {
  onUtterance: (blob: Blob) => void;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export function useVoiceCall({ onUtterance }: UseVoiceCallOptions) {
  const supported =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined" &&
    typeof AudioContext !== "undefined";

  const [inCall, setInCall] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);

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
  const busyRef = useRef(false); // paused while a reply is being fetched/played, so we don't hear ourselves
  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;

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
    if (streamRef.current) startSegmentRecorder();
  }, [startSegmentRecorder]);

  const pauseForReply = useCallback(() => {
    busyRef.current = true;
    stopSegmentRecorder();
  }, [stopSegmentRecorder]);

  const startCall = useCallback(async () => {
    if (!supported || inCall) return;
    setPermissionError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      busyRef.current = false;
      startSegmentRecorder();
      setInCall(true);

      const buf = new Uint8Array(analyser.fftSize);
      intervalRef.current = window.setInterval(() => {
        if (busyRef.current || !recorderRef.current) return;
        analyser.getByteTimeDomainData(buf);
        let sumSquares = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sumSquares += v * v;
        }
        const rms = Math.sqrt(sumSquares / buf.length);
        const now = performance.now();
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
      setPermissionError(
        err instanceof Error ? err.message : "Microphone access was denied or unavailable."
      );
      setInCall(false);
    }
  }, [supported, inCall, startSegmentRecorder, pauseForReply]);

  const endCall = useCallback(() => {
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
    setInCall(false);
  }, [stopSegmentRecorder]);

  return { supported, inCall, permissionError, startCall, endCall, pauseForReply, resumeListening };
}
