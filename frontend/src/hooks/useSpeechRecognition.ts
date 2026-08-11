import { useCallback, useEffect, useRef, useState } from "react";

// Minimal shape of the Web Speech API - not in lib.dom.d.ts by default.
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: { transcript: string };
}
interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}
interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onend: (() => void) | null;
}

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | undefined {
  if (typeof window === "undefined") return undefined;
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

interface UseSpeechRecognitionOptions {
  onFinalResult: (text: string) => void;
  onInterimResult?: (text: string) => void;
  lang?: string;
}

export function useSpeechRecognition({ onFinalResult, onInterimResult, lang = "en-US" }: UseSpeechRecognitionOptions) {
  const RecognitionCtor = getRecognitionCtor();
  const supported = Boolean(RecognitionCtor);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldRestartRef = useRef(false);
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    if (!RecognitionCtor) return;
    const recognition = new RecognitionCtor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = lang;

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0].transcript;
        if (result.isFinal) {
          onFinalResult(transcript.trim());
        } else {
          interim += transcript;
        }
      }
      if (interim) onInterimResult?.(interim.trim());
    };

    recognition.onerror = () => {
      // onend fires right after; restart decision happens there
    };

    recognition.onend = () => {
      if (shouldRestartRef.current) {
        try {
          recognition.start();
        } catch {
          // recognition was already starting - browser will settle on its own
        }
      } else {
        setIsListening(false);
      }
    };

    recognitionRef.current = recognition;
    return () => {
      shouldRestartRef.current = false;
      recognition.onresult = null;
      recognition.onend = null;
      recognition.onerror = null;
      recognition.stop();
    };
  }, [RecognitionCtor, lang, onFinalResult, onInterimResult]);

  const start = useCallback(() => {
    if (!recognitionRef.current) return;
    shouldRestartRef.current = true;
    try {
      recognitionRef.current.start();
      setIsListening(true);
    } catch {
      // already listening
    }
  }, []);

  const stop = useCallback(() => {
    shouldRestartRef.current = false;
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  return { supported, isListening, start, stop };
}
