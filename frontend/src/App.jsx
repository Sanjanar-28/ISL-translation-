import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

// =========================================================
// 49 SIGN VIDEO / TRANSLATION MAPPING
// =========================================================

const SIGN_ALIASES = {
  bird: "bird",
  black: "black",
  cat: "cat",
  cow: "cow",
  dog: "dog",
  fish: "fish",

  goodmorning: "goodmorning",
  "good morning": "goodmorning",

  grey: "grey",
  gray: "grey",

  hello: "hello",
  hi: "hello",
  namaste: "hello",
  namaskar: "hello",
  namaskara: "hello",

  youareperfect: "youareperfect",
  "you are perfect": "youareperfect",

  monsoon: "monsoon",
  afternoon: "afternoon",
  angry: "angry",
  bad: "bad",
  boy: "boy",

  eat: "eat",
  eating: "eat",

  friend: "friend",

  drinking: "drinking",
  drink: "drinking",

  girl: "girl",
  brother: "brother",
  good: "good",
  father: "father",
  evening: "evening",
  help: "help",
  mother: "mother",

  name: "NAME",

  night: "Night",
  music: "Music",

  nose: "nose",
  sleep: "sleep",
  sit: "sit",
  sorry: "sorry",
  stand: "stand",
  stop: "stop",
  student: "student",
  study: "study",
  teacher: "teacher",

  thankyou: "thankyou",
  "thank you": "thankyou",

  today: "today",

  tomorrow: "tommorow",
  tommorow: "tommorow",

  welcome: "welcome",
  work: "work",
  yesterday: "yesterday",

  teeth: "teeth",
  hand: "hand",
  write: "write",

  umbrella: "umberlla",
  umberlla: "umberlla",

  ring: "ring",
  power: "power",
};

// =========================================================
// LANGUAGE CODES
// =========================================================

const LANGUAGE_CODES = {
  English: "en-IN",
  Hindi: "hi-IN",
  Kannada: "kn-IN",
};

function App() {
  // =======================================================
  // NAVIGATION
  // =======================================================

  const [page, setPage] = useState("home");

  const [mode, setMode] = useState("sign");

  // =======================================================
  // USER / PROFILE
  // =======================================================

  const [userId, setUserId] = useState(
    () => localStorage.getItem("isl_user_id") || ""
  );

  const [profile, setProfile] = useState({
    name: "",
    email: "",
    phone: "",
    preferredLanguage: "English",
  });

  const [profileMessage, setProfileMessage] =
    useState("");

  const [profileLoading, setProfileLoading] =
    useState(false);

  // =======================================================
  // HISTORY
  // =======================================================

  const [history, setHistory] = useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  // =======================================================
  // LANGUAGE
  // =======================================================

  const [language, setLanguage] =
    useState("English");

  // =======================================================
  // TEXT / SPEECH
  // =======================================================

  const [textInput, setTextInput] =
    useState("");

  const [isListening, setIsListening] =
    useState(false);

  // =======================================================
  // CAMERA REFS
  // =======================================================

  const videoRef = useRef(null);

  const streamRef = useRef(null);

  const recorderRef = useRef(null);

  const recordedChunksRef = useRef([]);

  const fileInputRef = useRef(null);

  // =======================================================
  // CAMERA
  // =======================================================

  const [cameraActive, setCameraActive] =
    useState(false);

  const [cameraError, setCameraError] =
    useState("");

  const [videoMode, setVideoMode] =
    useState("none");

  const [uploadedVideoUrl, setUploadedVideoUrl] =
    useState(null);

  const [selectedFile, setSelectedFile] =
    useState(null);

  // =======================================================
  // PREDICTION
  // =======================================================

  const [prediction, setPrediction] =
    useState(null);

  const [isPredicting, setIsPredicting] =
    useState(false);

  const [predictionError, setPredictionError] =
    useState("");

  // =======================================================
  // MULTILINGUAL TRANSLATION
  // =======================================================

  const [translatedText, setTranslatedText] =
    useState("");

  const [translationLoading, setTranslationLoading] =
    useState(false);

  // =======================================================
  // TEXT → SIGN VIDEO
  // =======================================================

  const [signVideo, setSignVideo] =
    useState("");

  const [selectedSign, setSelectedSign] =
    useState("");

  const [signVideoError, setSignVideoError] =
    useState("");

  // =======================================================
  // INITIAL PROFILE LOAD
  // =======================================================

  useEffect(() => {
    if (userId) {
      loadProfile(userId);
      loadHistory(userId);
    }
  }, []);

  // =======================================================
  // CAMERA CLEANUP
  // =======================================================

  useEffect(() => {
    return () => {
      if (
        recorderRef.current &&
        recorderRef.current.state !== "inactive"
      ) {
        recorderRef.current.stop();
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());

        streamRef.current = null;
      }

      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl);
      }
    };
  }, [uploadedVideoUrl]);

  // =======================================================
  // CAMERA PREVIEW
  // =======================================================

  useEffect(() => {
    if (
      videoMode === "camera" &&
      videoRef.current &&
      streamRef.current
    ) {
      const video = videoRef.current;

      video.srcObject = streamRef.current;
      video.muted = true;
      video.playsInline = true;

      video.play().catch((error) => {
        console.error(
          "Camera preview error:",
          error
        );
      });
    }
  }, [videoMode]);

  // =======================================================
  // UPLOADED VIDEO PREVIEW
  // =======================================================

  useEffect(() => {
    if (
      videoMode === "upload" &&
      uploadedVideoUrl &&
      videoRef.current
    ) {
      const video = videoRef.current;

      video.pause();
      video.srcObject = null;
      video.src = uploadedVideoUrl;

      video.load();

      video.play().catch(() => {});
    }
  }, [videoMode, uploadedVideoUrl]);

  // =======================================================
  // RE-TRANSLATE WHEN LANGUAGE CHANGES
  // =======================================================

  useEffect(() => {
    if (
      prediction &&
      prediction.label
    ) {
      translatePrediction(
        prediction.label
      );
    }
  }, [language]);

  // =======================================================
  // LOAD PROFILE
  // =======================================================

  const loadProfile = async (id) => {
    if (!id) return;

    try {
      const response = await fetch(
        `${API_URL}/profile/${id}`
      );

      if (!response.ok) {
        return;
      }

      const data =
        await response.json();

      setProfile({
        name: data.name || "",
        email: data.email || "",
        phone: data.phone || "",
        preferredLanguage:
          data.preferredLanguage ||
          "English",
      });

      if (data.preferredLanguage) {
        setLanguage(
          data.preferredLanguage
        );
      }
    } catch (error) {
      console.error(
        "Profile loading error:",
        error
      );
    }
  };

  // =======================================================
  // SAVE PROFILE
  // =======================================================

  const saveProfile = async () => {
    if (!profile.name.trim()) {
      setProfileMessage(
        "Please enter your name."
      );
      return;
    }

    if (!profile.email.trim()) {
      setProfileMessage(
        "Please enter your email."
      );
      return;
    }

    try {
      setProfileLoading(true);
      setProfileMessage("");

      const response = await fetch(
        `${API_URL}/profile`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            userId: userId || null,
            name: profile.name,
            email: profile.email,
            phone: profile.phone,
            preferredLanguage:
              profile.preferredLanguage,
          }),
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Unable to save profile."
        );
      }

      const newUserId =
        data.userId;

      localStorage.setItem(
        "isl_user_id",
        newUserId
      );

      setUserId(newUserId);

      setProfileMessage(
        "Profile saved successfully."
      );

      await loadHistory(newUserId);
    } catch (error) {
      console.error(
        "Profile save error:",
        error
      );

      setProfileMessage(
        error.message ||
          "Unable to save profile."
      );
    } finally {
      setProfileLoading(false);
    }
  };

  // =======================================================
  // LOAD HISTORY
  // =======================================================

  const loadHistory = async (id) => {
    if (!id) return;

    try {
      setHistoryLoading(true);

      const response = await fetch(
        `${API_URL}/history/${id}`
      );

      if (!response.ok) {
        setHistory([]);
        return;
      }

      const data =
        await response.json();

      setHistory(
        Array.isArray(data)
          ? data
          : []
      );
    } catch (error) {
      console.error(
        "History loading error:",
        error
      );

      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  // =======================================================
  // SAVE HISTORY
  // =======================================================

  const saveHistory = async (
    predictionData
  ) => {
    if (!userId) {
      return;
    }

    if (
      !predictionData ||
      !predictionData.label
    ) {
      return;
    }

    try {
      await fetch(
        `${API_URL}/history`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            sign: predictionData.label,
            confidence:
              predictionData.confidence ||
              0,
          }),
        }
      );

      await loadHistory(userId);
    } catch (error) {
      console.error(
        "History save error:",
        error
      );
    }
  };

  // =======================================================
  // MULTILINGUAL SIGN TRANSLATION
  // =======================================================

  const translatePrediction = async (
    sign
  ) => {
    if (!sign) {
      return "";
    }

    try {
      setTranslationLoading(true);

      const response = await fetch(
        `${API_URL}/translate-sign`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            sign: sign,
            language: language,
          }),
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Translation failed."
        );
      }

      setTranslatedText(
        data.translatedText
      );

      return data.translatedText;
    } catch (error) {
      console.error(
        "Translation error:",
        error
      );

      setTranslatedText(
        sign
      );

      return sign;
    } finally {
      setTranslationLoading(false);
    }
  };

  // =======================================================
  // START CAMERA
  // =======================================================

  const startCamera = async () => {
    try {
      setCameraError("");
      setPredictionError("");
      setPrediction(null);
      setTranslatedText("");
      setSelectedFile(null);

      if (uploadedVideoUrl) {
        URL.revokeObjectURL(
          uploadedVideoUrl
        );

        setUploadedVideoUrl(null);
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());

        streamRef.current = null;
      }

      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            video: {
              width: {
                ideal: 1280,
              },
              height: {
                ideal: 720,
              },
              facingMode: "user",
            },
            audio: false,
          }
        );

      streamRef.current = stream;

      recordedChunksRef.current = [];

      let mimeType = "";

      if (
        MediaRecorder.isTypeSupported(
          "video/webm;codecs=vp8"
        )
      ) {
        mimeType =
          "video/webm;codecs=vp8";
      } else if (
        MediaRecorder.isTypeSupported(
          "video/webm"
        )
      ) {
        mimeType =
          "video/webm";
      }

      const recorder = mimeType
        ? new MediaRecorder(
            stream,
            {
              mimeType,
            }
          )
        : new MediaRecorder(
            stream
          );

      recorderRef.current =
        recorder;

      recorder.ondataavailable =
        (event) => {
          if (
            event.data &&
            event.data.size > 0
          ) {
            recordedChunksRef.current.push(
              event.data
            );
          }
        };

      recorder.onstop = async () => {
        const chunks =
          recordedChunksRef.current;

        if (!chunks.length) {
          setPredictionError(
            "No video frames were recorded."
          );

          return;
        }

        const blob = new Blob(
          chunks,
          {
            type:
              recorder.mimeType ||
              "video/webm",
          }
        );

        const recordedFile =
          new File(
            [blob],
            "live_sign.webm",
            {
              type: "video/webm",
            }
          );

        await predictLiveVideo(
          recordedFile
        );
      };

      recorder.start(100);

      setVideoMode("camera");
      setCameraActive(true);
    } catch (error) {
      console.error(
        "Camera error:",
        error
      );

      setCameraError(
        "Unable to access the camera. Please allow camera permission."
      );

      setCameraActive(false);
      setVideoMode("none");
    }
  };

  // =======================================================
  // STOP CAMERA
  // =======================================================

  const stopCamera = () => {
    setCameraActive(false);

    if (recorderRef.current) {
      const recorder =
        recorderRef.current;

      if (
        recorder.state !==
        "inactive"
      ) {
        recorder.stop();
      }

      recorderRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) =>
          track.stop()
        );

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();

      videoRef.current.srcObject =
        null;
    }

    setVideoMode("none");
  };

  // =======================================================
  // LIVE CNN-LSTM PREDICTION
  // =======================================================

  const predictLiveVideo = async (
    videoFile
  ) => {
    try {
      setPredictionError("");
      setPrediction(null);
      setTranslatedText("");
      setIsPredicting(true);

      const formData =
        new FormData();

      formData.append(
        "video",
        videoFile
      );

      const response =
        await fetch(
          `${API_URL}/predict-live-video`,
          {
            method: "POST",
            body: formData,
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Live prediction failed."
        );
      }

      setPrediction(data);

      await saveHistory(data);

      await translatePrediction(
        data.label
      );
    } catch (error) {
      console.error(
        "Live prediction error:",
        error
      );

      setPredictionError(
        error.message ||
          "Unable to connect to prediction server."
      );
    } finally {
      setIsPredicting(false);
    }
  };

  // =======================================================
  // UPLOAD VIDEO
  // =======================================================

  const handleUploadVideo = (
    event
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    setCameraError("");
    setPredictionError("");
    setPrediction(null);
    setTranslatedText("");

    if (
      !file.type.startsWith(
        "video/"
      )
    ) {
      setCameraError(
        "Please select a valid video file."
      );

      return;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) =>
          track.stop()
        );

      streamRef.current = null;
    }

    setCameraActive(false);

    if (uploadedVideoUrl) {
      URL.revokeObjectURL(
        uploadedVideoUrl
      );
    }

    const videoUrl =
      URL.createObjectURL(file);

    setSelectedFile(file);

    setUploadedVideoUrl(
      videoUrl
    );

    setVideoMode("upload");

    event.target.value = "";
  };

  // =======================================================
  // UPLOADED VIDEO PREDICTION
  // =======================================================

  const predictSign = async () => {
    if (!selectedFile) {
      setPredictionError(
        "Please upload a video before predicting."
      );

      return;
    }

    try {
      setPredictionError("");
      setPrediction(null);
      setTranslatedText("");
      setIsPredicting(true);

      const formData =
        new FormData();

      formData.append(
        "video",
        selectedFile
      );

      const response =
        await fetch(
          `${API_URL}/predict`,
          {
            method: "POST",
            body: formData,
          }
        );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Prediction failed."
        );
      }

      setPrediction(data);

      await saveHistory(data);

      await translatePrediction(
        data.label
      );
    } catch (error) {
      console.error(
        "Prediction error:",
        error
      );

      setPredictionError(
        error.message ||
          "Unable to connect to prediction server."
      );
    } finally {
      setIsPredicting(false);
    }
  };

  // =======================================================
  // SPEAK MULTILINGUAL RESULT
  // =======================================================

  const speakResult = async () => {
  if (!translatedText) {
    alert("No translated text available.");
    return;
  }

  try {
    const response = await fetch(`${API_URL}/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: translatedText,
        language: language,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("TTS error:", errorText);
      alert("Speech generation failed.");
      return;
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);

    const audio = new Audio(audioUrl);

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
    };

    await audio.play();
  } catch (error) {
    console.error("Speech error:", error);
    alert("Unable to play speech.");
  }
};
  // =======================================================
  // TEXT TO SPEECH
  // =======================================================

  const detectTextLanguage = (text) => {
    // Kannada Unicode range
    if (/[^\u0000-\u007F]/.test(text) &&
        /[\u0C80-\u0CFF]/.test(text)) {
      return "Kannada";
    }

    // Hindi / Devanagari Unicode range
    if (/[\u0900-\u097F]/.test(text)) {
      return "Hindi";
    }

    // Otherwise treat it as English
    return "English";
  };

  const speakText = async () => {
  const text = textInput.trim();

  if (!text) {
    alert("Please enter some text first.");
    return;
  }

  let detectedLanguage = "English";

  // Hindi / Devanagari
  if (/[\u0900-\u097F]/.test(text)) {
    detectedLanguage = "Hindi";
  }

  // Kannada
  else if (/[\u0C80-\u0CFF]/.test(text)) {
    detectedLanguage = "Kannada";
  }

  // English
  else {
    detectedLanguage = "English";
  }

  try {
    const response = await fetch(
      `${API_URL}/tts`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          language: detectedLanguage,
        }),
      }
    );

    if (!response.ok) {
      const errorData = await response.json();

      throw new Error(
        errorData.detail ||
          "Text-to-speech failed."
      );
    }

    const audioBlob =
      await response.blob();

    const audioUrl =
      URL.createObjectURL(audioBlob);

    const audio =
      new Audio(audioUrl);

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
    };

    await audio.play();

  } catch (error) {
    console.error(
      "TTS error:",
      error
    );

    alert(
      "Unable to generate speech: " +
        error.message
    );
  }
};

  // =======================================================
  // TEXT → SIGN
const convertToSign = (inputTextOverride = textInput) => {
  const rawInput =
    typeof inputTextOverride === "string"
      ? inputTextOverride
      : textInput;

  const input = rawInput
    .trim()
    .toLowerCase()
    .replace(/[.!?,;:]+$/g, "")
    .replace(/\s+/g, " ");

  console.log("Convert to Sign input:", input);

  setSignVideoError("");
  setSignVideo("");
  setSelectedSign("");

  if (!input) {
    setSignVideoError("Please enter some text.");
    return;
  }

  // Exact supported sign / phrase
  if (SIGN_ALIASES[input]) {
    const sign = SIGN_ALIASES[input];

    console.log("Matched ISL sign:", sign);

    setSelectedSign(sign);
    setSignVideo(
      `${API_URL}/sign-video/${encodeURIComponent(sign)}`
    );
    return;
  }

  // Find the first supported sign in a sentence
  const words = input
    .split(/\s+/)
    .map((word) =>
      word.replace(/[^a-zA-Z]/g, "")
    )
    .filter(Boolean);

  for (const word of words) {
    if (SIGN_ALIASES[word]) {
      const sign = SIGN_ALIASES[word];

      console.log("Matched ISL sign:", sign);

      setSelectedSign(sign);
      setSignVideo(
        `${API_URL}/sign-video/${encodeURIComponent(sign)}`
      );
      return;
    }
  }

  setSignVideoError(
    `No matching ISL sign found for "${rawInput}".`
  );
};


  // =======================================================
  // SPEECH → TEXT
  // =======================================================
const startSpeechRecognition = () => {
  const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    alert(
      "Speech recognition is not supported in this browser. Please use Google Chrome."
    );
    return;
  }

  const recognition =
    new SpeechRecognition();

  recognition.lang = "en-IN";
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    setIsListening(true);
  };

  recognition.onresult = (event) => {
    const text =
      event.results[0][0].transcript
        .trim()
        .toLowerCase()
        .replace(/[.!?,;:]+$/g, "")
        .replace(/\s+/g, " ");

    console.log("Speech recognized:", text);

    /*
     * Speech → English ISL label
     *
     * These are the 49 signs in your project.
     * English, Hindi and common Kannada/romanized
     * forms are mapped to the existing SIGN_ALIASES.
     */
    const spokenToEnglish = {

      // 1 Music
      "music": "Music",
      "संगीत": "Music",
      "sangeet": "Music",
      "sangeetha": "Music",

      // 2 NAME
      "name": "NAME",
      "my name": "NAME",
      "नाम": "NAME",
      "naam": "NAME",
      "hesaru": "NAME",

      // 3 Night
      "night": "Night",
      "रात": "Night",
      "raat": "Night",
      "रात्री": "Night",
      "ratri": "Night",

      // 4 afternoon
      "afternoon": "afternoon",
      "दोपहर": "afternoon",
      "dopahar": "afternoon",
      "madhyahna": "afternoon",

      // 5 angry
      "angry": "angry",
      "गुस्सा": "angry",
      "gussa": "angry",
      "kopa": "angry",
      "koppa": "angry",

      // 6 bad
      "bad": "bad",
      "बुरा": "bad",
      "bura": "bad",
      "kettadu": "bad",
      "ketta": "bad",

      // 7 bird
      "bird": "bird",
      "पक्षी": "bird",
      "pakshi": "bird",
      "hakki": "bird",
      "hakkī": "bird",

      // 8 black
      "black": "black",
      "काला": "black",
      "kala": "black",
      "kappu": "black",

      // 9 boy
      "boy": "boy",
      "लड़का": "boy",
      "ladka": "boy",
      "huduga": "boy",

      // 10 brother
      "brother": "brother",
      "भाई": "brother",
      "bhai": "brother",
      "anna": "brother",
      "tamma": "brother",

      // 11 cat
      "cat": "cat",
      "बिल्ली": "cat",
      "billi": "cat",
      "bekku": "cat",

      // 12 cow
      "cow": "cow",
      "गाय": "cow",
      "gaay": "cow",
      "hasu": "cow",

      // 13 dog
      "dog": "dog",
      "कुत्ता": "dog",
      "kutta": "dog",
      "naayi": "dog",
      "nayi": "dog",

      // 14 drinking
      "drink": "drinking",
      "drinking": "drinking",
      "पीना": "drinking",
      "peena": "drinking",
      "kudi": "drinking",
      "kudiyodu": "drinking",

      // 15 eat
      "eat": "eat",
      "eating": "eat",
      "खाना": "eat",
      "khana": "eat",
      "tinnu": "eat",
      "tinnodu": "eat",

      // 16 evening
      "evening": "evening",
      "शाम": "evening",
      "shaam": "evening",
      "sanje": "evening",

      // 17 father
      "father": "father",
      "पिता": "father",
      "pita": "father",
      "पापा": "father",
      "papa": "father",
      "appa": "father",

      // 18 fish
      "fish": "fish",
      "मछली": "fish",
      "machli": "fish",
      "meenu": "fish",

      // 19 friend
      "friend": "friend",
      "दोस्त": "friend",
      "dost": "friend",
      "snehita": "friend",
      "geleya": "friend",
      "geley": "friend",

      // 20 girl
      "girl": "girl",
      "लड़की": "girl",
      "ladki": "girl",
      "hudugi": "girl",

      // 21 good
      "good": "good",
      "अच्छा": "good",
      "accha": "good",
      "chennagide": "good",
      "olledu": "good",

      // 22 good morning
      "good morning": "goodmorning",
      "goodmorning": "goodmorning",
      "सुप्रभात": "goodmorning",
      "suprabhat": "goodmorning",
      "shubhodaya": "goodmorning",

      // 23 grey
      "grey": "grey",
      "gray": "grey",
      "ग्रे": "grey",

      // 24 hand
      "hand": "hand",
      "हाथ": "hand",
      "haath": "hand",
      "kai": "hand",

      // 25 hello
      "hello": "hello",
      "hi": "hello",
      "namaste": "hello",
      "namaskar": "hello",
      "namaskara": "hello",
      "नमस्ते": "hello",
      "नमस्कार": "hello",
      "ನಮಸ್ಕಾರ": "hello",

      // 26 help
      "help": "help",
      "मदद": "help",
      "madad": "help",
      "sahaya": "help",
      "sahaya maadi": "help",

      // 27 monsoon
      "monsoon": "monsoon",
      "मानसून": "monsoon",
      "mansoon": "monsoon",
      "ಮಳೆಗಾಲ": "monsoon",
      "malegala": "monsoon",

      // 28 mother
      "mother": "mother",
      "माँ": "mother",
      "माता": "mother",
      "maa": "mother",
      "amma": "mother",

      // 29 nose
      "nose": "nose",
      "नाक": "nose",
      "naak": "nose",
      "mooku": "nose",

      // 30 power
      "power": "power",
      "शक्ति": "power",
      "shakti": "power",

      // 31 ring
      "ring": "ring",
      "अंगूठी": "ring",
      "anguthi": "ring",
      "ungura": "ring",

      // 32 sit
      "sit": "sit",
      "बैठो": "sit",
      "baitho": "sit",
      "kootko": "sit",
      "koothko": "sit",

      // 33 sleep
      "sleep": "sleep",
      "सोना": "sleep",
      "sona": "sleep",
      "nidre": "sleep",

      // 34 sorry
      "sorry": "sorry",
      "सॉरी": "sorry",
      "maaf": "sorry",
      "क्षमा": "sorry",
      "kshamisi": "sorry",

      // 35 stand
      "stand": "stand",
      "खड़े हो": "stand",
      "khade ho": "stand",
      "nillu": "stand",

      // 36 stop
      "stop": "stop",
      "रुको": "stop",
      "ruko": "stop",
      "nillisi": "stop",

      // 37 student
      "student": "student",
      "विद्यार्थी": "student",
      "vidyarthi": "student",
      "studentu": "student",

      // 38 study
      "study": "study",
      "पढ़ाई": "study",
      "padhai": "study",
      "odhu": "study",
      "oduvudu": "study",

      // 39 teacher
      "teacher": "teacher",
      "शिक्षक": "teacher",
      "shikshak": "teacher",
      "guru": "teacher",
      "shikshaka": "teacher",

      // 40 teeth
      "teeth": "teeth",
      "दाँत": "teeth",
      "daant": "teeth",
      "hallugalu": "teeth",

      // 41 thank you
      "thank you": "thankyou",
      "thankyou": "thankyou",
      "धन्यवाद": "thankyou",
      "dhanyavaad": "thankyou",
      "dhanyavad": "thankyou",
      "dhanyavada": "thankyou",

      // 42 today
      "today": "today",
      "आज": "today",
      "aaj": "today",
      "ivattu": "today",

      // 43 tomorrow
      "tomorrow": "tommorow",
      "tommorow": "tommorow",
      "कल": "tommorow",
      "naale": "tommorow",

      // 44 umbrella
      "umbrella": "umberlla",
      "umberlla": "umberlla",
      "छाता": "umberlla",
      "chhata": "umberlla",

      // 45 welcome
      "welcome": "welcome",
      "स्वागत": "welcome",
      "swagat": "welcome",
      "suswagata": "welcome",

      // 46 work
      "work": "work",
      "काम": "work",
      "kaam": "work",
      "kelasa": "work",

      // 47 write
      "write": "write",
      "लिखना": "write",
      "likhna": "write",
      "bare": "write",
      "bareyodu": "write",

      // 48 yesterday
      "yesterday": "yesterday",
      "कल": "yesterday",
      "ninne": "yesterday",

      // 49 you are perfect
      "you are perfect": "youareperfect",
      "youareperfect": "youareperfect",
      "आप परफेक्ट हैं": "youareperfect",
      "neevu perfect": "youareperfect"
    };

    const normalizedText =
      spokenToEnglish[text] ||
      SIGN_ALIASES[text] ||
      text;

    console.log(
      "Translated to English:",
      normalizedText
    );

    // Show the English form in textarea.
    // Do NOT play the video automatically.
    setTextInput(normalizedText);
    setIsListening(false);
  };

  recognition.onerror = (event) => {
    console.error(
      "Speech recognition error:",
      event.error
    );

    setIsListening(false);

    if (event.error === "not-allowed") {
      alert(
        "Please allow microphone access."
      );
    } else {
      alert(
        "Could not understand the speech. Please try again."
      );
    }
  };

  recognition.onend = () => {
    setIsListening(false);
  };

  recognition.start();
};


  // =======================================================
  // HOME PAGE
  // =======================================================

  const renderHome = () => {
    return (
      <>
        {/* HERO */}

        <section className="hero">
          <p className="hero-tag">
            AI-POWERED COMMUNICATION
          </p>

          <h1>
            Bridging{" "}
            <span>
              Sign & Speech
            </span>
          </h1>

          <p className="hero-description">
            <br></br>
            Communicate effortlessly
            with real-time Indian
            Sign Language recognition
            and multilingual
            translation.
          </p>
        </section>

        {/* MODE SWITCH */}

        <div className="mode-switch">

          <button
            className={
              mode === "sign"
                ? "mode-button active-mode"
                : "mode-button"
            }
            onClick={() =>
              setMode("sign")
            }
          >
            🤟 Sign to Text
          </button>

          <button
            className={
              mode === "text"
                ? "mode-button active-mode"
                : "mode-button"
            }
            onClick={() =>
              setMode("text")
            }
          >
            💬 Text to Sign
          </button>

        </div>

        {/* =================================================
            SIGN → TEXT
        ================================================= */}

        {mode === "sign" && (
          <div className="content-grid">

            {/* CAMERA CARD */}

            <section className="card sign-card">

              <h2>
                Sign Recognition
              </h2>

              <div className="camera-preview">

                {videoMode ===
                  "none" && (
                  <div className="camera-placeholder">

                    <div className="camera-icon">
                      📹
                    </div>

                    <p>
                      Camera Preview
                    </p>

                  </div>
                )}

                {videoMode !==
                  "none" && (
                  <video
                    ref={videoRef}
                    className="preview-video"
                    autoPlay
                    playsInline
                    muted={
                      videoMode ===
                      "camera"
                    }
                    controls={
                      videoMode ===
                      "upload"
                    }
                  />
                )}

              </div>

              {cameraError && (
                <p className="camera-error">
                  {cameraError}
                </p>
              )}

              {predictionError && (
                <p className="camera-error">
                  {predictionError}
                </p>
              )}

              <div className="button-group">

                {!cameraActive ? (
                  <button
                    className="primary-button"
                    onClick={
                      startCamera
                    }
                  >
                    📹 Start Camera
                  </button>
                ) : (
                  <button
                    className="danger-button"
                    onClick={
                      stopCamera
                    }
                  >
                    ⏹ Stop Camera
                  </button>
                )}

                <button
                  className="secondary-button"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  📁 Upload Video
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  style={{
                    display: "none",
                  }}
                  onChange={
                    handleUploadVideo
                  }
                />

              </div>

              {videoMode ===
                "upload" && (
                <button
                  className="primary-button"
                  onClick={
                    predictSign
                  }
                  disabled={
                    isPredicting
                  }
                >
                  {isPredicting
                    ? "⏳ Predicting..."
                    : "🤟 Predict Sign"}
                </button>
              )}

            </section>

            {/* PREDICTION CARD */}

            <section className="card result-card">

              <h2>
                Prediction
              </h2>

              <div className="prediction-box">

                <h3>
                  {isPredicting
                    ? "PROCESSING..."
                    : translationLoading
                    ? "TRANSLATING..."
                    : translatedText ||
                      prediction?.label ||
                      "WAITING..."}
                </h3>

                {prediction && (
                  <>
                    <p className="prediction-success">
                      Prediction Complete
                    </p>

                    <p>
                      ISL Sign:{" "}
                      <strong>
                        {prediction.label}
                      </strong>
                    </p>
                  </>
                )}

                <p className="confidence">
                  Confidence:{" "}
                  {prediction
                    ? `${(
                        prediction.confidence *
                        100
                      ).toFixed(2)}%`
                    : "—"}
                </p>

                <select
                  className="language-select"
                  value={language}
                  onChange={(e) =>
                    setLanguage(
                      e.target.value
                    )
                  }
                >
                  <option value="English">
                    English
                  </option>

                  <option value="Hindi">
                    Hindi
                  </option>

                  <option value="Kannada">
                    Kannada
                  </option>
                </select>

                <button
                  className="speak-button"
                  onClick={
                    speakResult
                  }
                >
                  🔊 Speak
                </button>

              </div>

            </section>

          </div>
        )}

        {/* =================================================
            TEXT → SIGN
        ================================================= */}

        {mode === "text" && (
          <section className="text-sign-container card">

            <h2>
              Text to Sign
            </h2>

            <p>
              Enter a sign name or a
              sentence containing one
              of the supported ISL signs.
            </p>

<textarea
  className="text-input"
  placeholder="Example: hello"
  value={textInput}
  onChange={(e) => setTextInput(e.target.value)}
/>

<div className="button-group">

  <button
    className="primary-button"
    onClick={convertToSign}
  >
    🤟 Convert to Sign
  </button>

  <button
    className="secondary-button"
    onClick={startSpeechRecognition}
  >
    {isListening
      ? "🎤 Listening..."
      : "🎤 Speak"}
  </button>

</div>

            {signVideoError && (
              <p className="camera-error">
                {signVideoError}
              </p>
            )}

            {signVideo && (
              <div
                className="sign-video-container"
                style={{
                  marginTop:
                    "25px",
                  textAlign:
                    "center",
                }}
              >

                <h3>
                  ISL Sign:{" "}
                  {selectedSign}
                </h3>

                <video
                  key={signVideo}
                  src={signVideo}
                  controls
                  autoPlay
                  playsInline
                  style={{
                    width: "100%",
                    maxWidth:
                      "700px",
                    borderRadius:
                      "12px",
                    marginTop:
                      "15px",
                  }}
                />

                <p>
                  Showing the
                  reference video
                  for the{" "}
                  <strong>
                    {selectedSign}
                  </strong>{" "}
                  sign.
                </p>

              </div>
            )}

          </section>
        )}

      </>
    );
  };

  // =======================================================
  // HISTORY PAGE
  // =======================================================

  const renderHistory = () => {
    return (
      <section className="card">

        <h2>
          Prediction History
        </h2>

        {!userId ? (
          <p>
            Create your profile first
            to save prediction history.
          </p>
        ) : historyLoading ? (
          <p>
            Loading history...
          </p>
        ) : history.length === 0 ? (
          <p>
            No prediction history yet.
          </p>
        ) : (
          <div
            style={{
              overflowX:
                "auto",
            }}
          >

            <table
              style={{
                width: "100%",
                borderCollapse:
                  "collapse",
              }}
            >

              <thead>

                <tr>

                  <th
                    style={{
                      padding:
                        "12px",
                      textAlign:
                        "left",
                    }}
                  >
                    Sign
                  </th>

                  <th
                    style={{
                      padding:
                        "12px",
                      textAlign:
                        "left",
                    }}
                  >
                    Confidence
                  </th>

                  <th
                    style={{
                      padding:
                        "12px",
                      textAlign:
                        "left",
                    }}
                  >
                    Date
                  </th>

                </tr>

              </thead>

              <tbody>

                {history.map(
                  (
                    item,
                    index
                  ) => (
                    <tr
                      key={
                        item._id ||
                        index
                      }
                    >

                      <td
                        style={{
                          padding:
                            "12px",
                        }}
                      >
                        {item.sign}
                      </td>

                      <td
                        style={{
                          padding:
                            "12px",
                        }}
                      >
                        {(
                          (item.confidence ||
                            0) *
                          100
                        ).toFixed(2)}
                        %
                      </td>

                      <td
                        style={{
                          padding:
                            "12px",
                        }}
                      >
                        {item.timestamp
                          ? new Date(
                              item.timestamp
                            ).toLocaleString()
                          : "—"}
                      </td>

                    </tr>
                  )
                )}

              </tbody>

            </table>

          </div>
        )}

      </section>
    );
  };

  // =======================================================
  // PROFILE PAGE
  // =======================================================

  const renderProfile = () => {
    return (
      <section className="card">

        <h2>
          Profile
        </h2>

        <div
          style={{
            maxWidth:
              "650px",
          }}
        >

          <label>
            Name
          </label>

          <input
            type="text"
            value={profile.name}
            onChange={(e) =>
              setProfile({
                ...profile,
                name: e.target.value,
              })
            }
            placeholder="Enter your name"
            style={{
              width:
                "100%",
              marginBottom:
                "15px",
              padding:
                "12px",
              boxSizing:
                "border-box",
            }}
          />

          <label>
            Email
          </label>

          <input
            type="email"
            value={profile.email}
            onChange={(e) =>
              setProfile({
                ...profile,
                email: e.target.value,
              })
            }
            placeholder="Enter your email"
            style={{
              width:
                "100%",
              marginBottom:
                "15px",
              padding:
                "12px",
              boxSizing:
                "border-box",
            }}
          />

          <label>
            Phone
          </label>

          <input
            type="tel"
            value={profile.phone}
            onChange={(e) =>
              setProfile({
                ...profile,
                phone: e.target.value,
              })
            }
            placeholder="Enter your phone number"
            style={{
              width:
                "100%",
              marginBottom:
                "15px",
              padding:
                "12px",
              boxSizing:
                "border-box",
            }}
          />

          <label>
            Preferred Language
          </label>

          <select
            value={
              profile.preferredLanguage
            }
            onChange={(e) => {

              setProfile({
                ...profile,
                preferredLanguage:
                  e.target.value,
              });

              setLanguage(
                e.target.value
              );

            }}
            style={{
              width:
                "100%",
              marginBottom:
                "20px",
              padding:
                "12px",
            }}
          >

            <option value="English">
              English
            </option>

            <option value="Hindi">
              Hindi
            </option>

            <option value="Kannada">
              Kannada
            </option>

          </select>

          <button
            className="primary-button"
            onClick={
              saveProfile
            }
            disabled={
              profileLoading
            }
          >
            {profileLoading
              ? "Saving..."
              : "Save Profile"}
          </button>

          {profileMessage && (
            <p
              style={{
                marginTop:
                  "15px",
              }}
            >
              {profileMessage}
            </p>
          )}

          {userId && (
            <p
              style={{
                marginTop:
                  "20px",
                fontSize:
                  "12px",
                opacity:
                  0.6,
              }}
            >
              User ID:{" "}
              {userId}
            </p>
          )}

        </div>

      </section>
    );
  };

  // =======================================================
  // MAIN RENDER
  // =======================================================

  return (
    <div className="app">

      {/* =================================================
          NAVBAR
      ================================================= */}

      <header className="navbar">

        <div
          className="logo"
          onClick={() =>
            setPage("home")
          }
          style={{
            cursor:
              "pointer",
          }}
        >

          <span className="logo-icon">
            🤟
          </span>

          <span>
            ISL ASSISTANT
          </span>

        </div>

        <nav className="nav-links">

          <button
            className={
              page === "home"
                ? "nav-link active"
                : "nav-link"
            }
            onClick={() =>
              setPage("home")
            }
          >
            Home
          </button>

          <button
            className={
              page === "history"
                ? "nav-link active"
                : "nav-link"
            }
            onClick={() => {

              setPage(
                "history"
              );

              if (userId) {
                loadHistory(
                  userId
                );
              }

            }}
          >
            History
          </button>

          <button
            className="profile-button"
            onClick={() =>
              setPage(
                "profile"
              )
            }
            title="Profile"
          >
            👤
          </button>

        </nav>

      </header>

      {/* =================================================
          MAIN
      ================================================= */}

      <main className="main-container">

        {page === "home" &&
          renderHome()}

        {page === "history" &&
          renderHistory()}

        {page === "profile" &&
          renderProfile()}

      </main>

      {/* =================================================
          FOOTER
      ================================================= */}

      <footer className="footer">

        <p>
          ISL Assistant •
          AI-Powered
          Communication
        </p>

      </footer>

    </div>
  );
}

export default App;