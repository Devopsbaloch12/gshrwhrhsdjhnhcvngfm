import { useCallback, useEffect, useRef } from "react";
import { useMotionValue, type MotionValue } from "framer-motion";

// Plays the assistant's reply and reports its live amplitude.
//
// The amplitude is a MotionValue rather than React state on purpose: it updates every
// animation frame, and pushing that through setState would re-render the whole section
// ~60 times a second. Framer reads a MotionValue outside the React render cycle.
export function useAudioPlayer(onEnded?: () => void): {
  play: (url: string) => void;
  stop: () => void;
  level: MotionValue<number>;
} {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const onEndedRef = useRef(onEnded);
  onEndedRef.current = onEnded;

  const level = useMotionValue(0);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  // Blob URLs live until explicitly revoked; every reply used to leak one for the
  // lifetime of the page.
  const currentUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const audio = new Audio();
    audio.crossOrigin = "anonymous";
    audioRef.current = audio;

    // "ended" is not the only way playback stops mattering. A decode failure or a
    // blocked autoplay used to leave the caller waiting on a callback that never came,
    // which stranded the UI on "speaking" with a silent orb.
    const settle = () => {
      // Stop metering first, so the orb settles instead of freezing on the last frame's
      // amplitude once playback is over.
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      level.set(0);
      onEndedRef.current?.();
    };
    audio.addEventListener("ended", settle);
    audio.addEventListener("error", settle);

    return () => {
      audio.removeEventListener("ended", settle);
      audio.removeEventListener("error", settle);
      audio.pause();
      if (currentUrlRef.current) URL.revokeObjectURL(currentUrlRef.current);
      currentUrlRef.current = null;
    };
  }, [level]);

  const stopMeter = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    level.set(0);
  }, [level]);

  // Wiring the element through an AnalyserNode is what makes the orb move in time with
  // the actual reply instead of running a canned animation that looks identical whether
  // audio is playing, silent, or failed to load.
  const ensureAnalyser = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || analyserRef.current) return analyserRef.current;
    try {
      const ctx = new AudioContext();
      const source = ctx.createMediaElementSource(audio);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      // Must also reach the speakers - a MediaElementSource routed only into an
      // analyser plays silently.
      analyser.connect(ctx.destination);
      ctxRef.current = ctx;
      analyserRef.current = analyser;
      return analyser;
    } catch {
      // No Web Audio (or the element is already bound to another context): fall back to
      // plain playback with a flat level. Sound still works; the orb just idles.
      return null;
    }
  }, []);

  const play = useCallback(
    (url: string) => {
      const audio = audioRef.current;
      if (!audio) return;

      const previous = currentUrlRef.current;
      currentUrlRef.current = url;
      audio.src = url;
      if (previous) URL.revokeObjectURL(previous);

      const analyser = ensureAnalyser();
      // Contexts created before a user gesture start suspended; without this the
      // analyser reports silence and nothing is heard.
      void ctxRef.current?.resume().catch(() => {});

      void audio
        .play()
        .then(() => {
          if (!analyser) return;
          const buf = new Uint8Array(analyser.fftSize);
          const tick = () => {
            analyser.getByteTimeDomainData(buf);
            let sumSquares = 0;
            for (let i = 0; i < buf.length; i++) {
              const v = (buf[i] - 128) / 128;
              sumSquares += v * v;
            }
            level.set(Math.sqrt(sumSquares / buf.length));
            rafRef.current = requestAnimationFrame(tick);
          };
          rafRef.current = requestAnimationFrame(tick);
        })
        .catch(() => {
          // Autoplay blocked, or the blob wouldn't decode. Report it as finished so the
          // caller returns to listening rather than waiting forever on "speaking".
          stopMeter();
          onEndedRef.current?.();
        });
    },
    [ensureAnalyser, level, stopMeter]
  );

  const stop = useCallback(() => {
    const audio = audioRef.current;
    stopMeter();
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
  }, [stopMeter]);

  useEffect(() => stopMeter, [stopMeter]);

  return { play, stop, level };
}
