import { useCallback, useEffect, useRef } from "react";

export function useAudioPlayer(onEnded?: () => void) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const onEndedRef = useRef(onEnded);
  onEndedRef.current = onEnded;

  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;
    const handleEnded = () => onEndedRef.current?.();
    audio.addEventListener("ended", handleEnded);
    return () => {
      audio.removeEventListener("ended", handleEnded);
      audio.pause();
    };
  }, []);

  const play = useCallback((url: string) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.src = url;
    void audio.play();
  }, []);

  const stop = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
  }, []);

  return { play, stop };
}
