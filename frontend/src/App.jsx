import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [mode, setMode] = useState("sign");

  // Kept only for UI dropdown appearance.
  // Multilingual translation logic is not implemented here.
  const [language, setLanguage] = useState("English");
  const [textInput, setTextInput] = useState("");
const [isListening, setIsListening] = useState(false);
const LANGUAGE_CODES = {
  English: "en-IN",
  Hindi: "hi-IN",
  Kannada: "kn-IN",
};
const TTS_LANGUAGE_CODES = {
  English: "en-IN",
  Hindi: "hi-IN",
  Kannada: "kn-IN",
};

  // none = no preview
  // camera = webcam active
  // upload = uploaded video active
  const [videoMode, setVideoMode] = useState("none");
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState(null);

  // Selected actual video file for backend prediction
  const [selectedFile, setSelectedFile] = useState(null);

  // Prediction states
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  // =========================================================
  // CLEANUP ON CLOSE
  // =========================================================

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }

      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl);
      }
    };
  }, [uploadedVideoUrl]);

  // =========================================================
  // LOAD UPLOADED VIDEO
  // =========================================================

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

      video.play().catch((error) => {
        console.log("Autoplay blocked:", error);
      });
    }
  }, [videoMode, uploadedVideoUrl]);

  // =========================================================
  // START CAMERA
  // =========================================================

  const startCamera = async () => {
    try {
      setCameraError("");
      setPredictionError("");
      setPrediction(null);
      setSelectedFile(null);

      // Remove old uploaded video
      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl);
        setUploadedVideoUrl(null);
      }

      // Stop old camera stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;

      setVideoMode("camera");
      setCameraActive(true);

      setTimeout(() => {
        if (videoRef.current) {
          const video = videoRef.current;

          video.pause();
          video.removeAttribute("src");
          video.load();

          video.srcObject = stream;

          video.play().catch((error) => {
            console.error("Camera play error:", error);
          });
        }
      }, 100);
    } catch (error) {
      console.error("Camera error:", error);

      setCameraError(
        "Unable to access the camera. Please allow camera permission."
      );

      setCameraActive(false);
      setVideoMode("none");
    }
  };

  // =========================================================
  // STOP CAMERA
  // =========================================================

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setVideoMode("none");
  };

  // =========================================================
  // UPLOAD VIDEO
  // =========================================================

  const handleUploadVideo = (event) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setCameraError("");
    setPredictionError("");
    setPrediction(null);

    // Check video file
    if (!file.type.startsWith("video/")) {
      setCameraError("Please select a valid video file.");
      return;
    }

    // Stop webcam first
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });

      streamRef.current = null;
    }

    setCameraActive(false);

    // Remove previous uploaded URL
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl);
    }

    // Create local preview URL
    const videoUrl = URL.createObjectURL(file);

    console.log("Selected video:", file.name);

    // Save actual file for backend
    setSelectedFile(file);

    // Save URL for preview
    setUploadedVideoUrl(videoUrl);
    setVideoMode("upload");

    // Allow selecting same file again
    event.target.value = "";
  };

  // =========================================================
  // PREDICT SIGN
  // =========================================================

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
      setIsPredicting(true);

      const formData = new FormData();

      formData.append(
        "video",
        selectedFile
      );

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      setPrediction(data);

      console.log(
        "Prediction result:",
        data
      );
    } catch (error) {
      console.error(
        "Prediction error:",
        error
      );

      setPredictionError(
        error.message ||
        "Unable to connect to the prediction server."
      );
    } finally {
      setIsPredicting(false);
    }
  };

  // =========================================================
  // SPEAK RESULT
  // =========================================================

  const speakResult = () => {
    if (!prediction || !prediction.label) {
      alert("Please predict a sign first.");
      return;
    }

    if (!("speechSynthesis" in window)) {
      alert("Speech is not supported by this browser.");
      return;
    }

    window.speechSynthesis.cancel();

    // Speak the direct ML model output.
    // No multilingual translation logic here.
    const speech = new SpeechSynthesisUtterance(
      prediction.label
    );

    speech.lang = "en-US";
    speech.rate = 0.85;
    speech.pitch = 1;

    window.speechSynthesis.speak(speech);
  };
// =========================================================
// TEXT TO SPEECH
// =========================================================

const speakText = async () => {
  if (!textInput.trim()) {
    alert("Please enter or speak some text first.");
    return;
  }

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/tts",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: textInput,
          language: language,
        }),
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail || "Text-to-speech failed."
      );
    }

    const audioBlob = await response.blob();

    const audioUrl = URL.createObjectURL(
      audioBlob
    );

    const audio = new Audio(audioUrl);

    audio.play();

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
    };

  } catch (error) {
    console.error(
      "Text-to-speech error:",
      error
    );

    alert(
      "Unable to generate speech: " +
      error.message
    );
  }
};
  // =========================================================
  // FORMAT CONFIDENCE
  // =========================================================

  const confidenceText = prediction
    ? `${(prediction.confidence * 100).toFixed(2)}%`
    : "—";
// =========================================================
// SPEECH TO TEXT
// =========================================================

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

  const recognition = new SpeechRecognition();

  recognition.lang = LANGUAGE_CODES[language];
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    setIsListening(true);
  };

  recognition.onresult = (event) => {
    const text =
      event.results[0][0].transcript;

    setTextInput(text);
    setIsListening(false);
  };

  recognition.onerror = (event) => {
    console.error(
      "Speech recognition error:",
      event.error
    );

    setIsListening(false);
  };

  recognition.onend = () => {
    setIsListening(false);
  };

  recognition.start();
};
  // =========================================================
  // RENDER
  // =========================================================

  return (
    <div className="app">

      {/* HEADER */}

      <header className="navbar">
        <div className="logo">
          <span className="logo-icon">🤟</span>
          <span>ISL ASSISTANT</span>
        </div>

        <nav className="nav-links">
          <button className="nav-link active">
            Home
          </button>

          <button className="nav-link">
            History
          </button>

          <button className="profile-button">
            👤
          </button>
        </nav>
      </header>

      <main className="main-container">

        {/* HERO */}

        <section className="hero">
          <p className="hero-tag">
            AI-POWERED COMMUNICATION
          </p>

          <h1>
            Bridging <span>Sign & Speech</span>
            <br />
          </h1>

          <p className="hero-description">
            <br />
            Communicate effortlessly with real-time Indian
            Sign Language recognition and multilingual
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
            onClick={() => setMode("sign")}
          >
            🤟 Sign to Text
          </button>

          <button
            className={
              mode === "text"
                ? "mode-button active-mode"
                : "mode-button"
            }
            onClick={() => setMode("text")}
          >
            💬 Text to Sign
          </button>
        </div>

        {/* SIGN TO TEXT */}

        {mode === "sign" && (
          <div className="content-grid">

            {/* CAMERA CARD */}

            <section className="card sign-card">
              <h2>Sign Recognition</h2>

              <div className="camera-preview">

                {/* NO VIDEO */}

                {videoMode === "none" && (
                  <div className="camera-placeholder">
                    <div className="camera-icon">
                      📹
                    </div>

                    <p>Camera Preview</p>
                  </div>
                )}

                {/* VIDEO */}

                {videoMode !== "none" && (
                  <video
                    ref={videoRef}
                    className={
                      videoMode === "camera"
                        ? "preview-video webcam-preview"
                        : "preview-video uploaded-preview"
                    }
                    autoPlay
                    playsInline
                    muted={videoMode === "camera"}
                    controls={videoMode === "upload"}
                  />
                )}
              </div>

              {/* ERROR */}

              {cameraError && (
                <p className="camera-error">
                  {cameraError}
                </p>
              )}

              {/* PREDICTION ERROR */}

              {predictionError && (
                <p className="camera-error">
                  {predictionError}
                </p>
              )}

              {/* BUTTONS */}

              <div className="camera-buttons">

                <button
                  className="primary-button"
                  onClick={
                    cameraActive
                      ? stopCamera
                      : startCamera
                  }
                >
                  {cameraActive
                    ? "Stop Camera"
                    : "Start Camera"}
                </button>

                <button
                  className="secondary-button"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  Upload Video
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp4,.webm,.mov,.avi,video/mp4,video/webm,video/quicktime,video/x-msvideo"
                  hidden
                  onChange={handleUploadVideo}
                />
              </div>

              {/* PREDICT BUTTON */}

              <button
                className="primary-button predict-button"
                onClick={predictSign}
                disabled={
                  !selectedFile ||
                  isPredicting
                }
              >
                {isPredicting
                  ? "Predicting..."
                  : "✨ Predict Sign"}
              </button>
            </section>

            {/* RESULT CARD */}

            <section className="card result-card">

              <h2>Recognition Result</h2>

              <div className="result-icon">
                {prediction ? "🤟" : "✋"}
              </div>

              <br />

              <h3>
  {prediction
    ? prediction.label.toUpperCase()
    : isPredicting
      ? "ANALYZING..."
      : "WAITING..."}
</h3>

{prediction && (
  <p className="prediction-success">
    Prediction Complete
  </p>
)}

<p className="confidence">
  Confidence: {confidenceText}
</p>

              {/* UI kept unchanged for future Member 2 integration */}

              <select
                className="language-select"
                value={language}
                onChange={(event) =>
                  setLanguage(event.target.value)
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
                onClick={speakResult}
              >
                🔊 Speak
              </button>

            </section>

          </div>
        )}

        {/* TEXT TO SIGN */}

        {mode === "text" && (
          <section className="text-sign-container card">

            <h2>Text to Sign</h2>

            <p>
              Enter text to convert it into
              Indian Sign Language.
            </p>

            <textarea
  className="text-input"
  placeholder="Enter your message here..."
  value={textInput}
  onChange={(event) =>
    setTextInput(event.target.value)
  }
/>

            <select
  className="language-select"
  value={language}
  onChange={(event) =>
    setLanguage(event.target.value)
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
  className="secondary-button"
  onClick={speakText}
>
  🔊 Speak Text
</button>

<button className="primary-button">
  Convert to Sign
</button>

<button
  className="secondary-button"
  onClick={startSpeechRecognition}
>
  {isListening
    ? "🎤 Listening..."
    : "🎤 Speak"}
</button>

<button className="primary-button">
  Convert to Sign
</button>

          </section>
        )}

      </main>

      {/* FOOTER */}

      <footer className="footer">
        <p>
          ISL Assistant • AI-Powered Communication
        </p>
      </footer>

    </div>
  );
}

export default App;