import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [mode, setMode] = useState("sign");

  const [language, setLanguage] = useState("English");
  const [textInput, setTextInput] = useState("");
  const [isListening, setIsListening] = useState(false);

  const LANGUAGE_CODES = {
    English: "en-IN",
    Hindi: "hi-IN",
    Kannada: "kn-IN",
  };

  const [videoMode, setVideoMode] = useState("none");
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  // =========================================================
  // CLEANUP
  // =========================================================

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
    };
  }, []);

  // =========================================================
  // ATTACH CAMERA STREAM AFTER VIDEO ELEMENT RENDERS
  // =========================================================

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

      video
        .play()
        .then(() => {
          console.log("Camera preview started.");
        })
        .catch((error) => {
          console.error("Video play error:", error);
        });
    }
  }, [videoMode]);

  // =========================================================
  // START CAMERA + RECORDING
  // =========================================================

  const startCamera = async () => {
    try {
      setCameraError("");
      setPredictionError("");
      setPrediction(null);
      setSelectedFile(null);

      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl);
        setUploadedVideoUrl(null);
      }

      // Stop any previous stream
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => track.stop());

        streamRef.current = null;
      }

      // Get webcam
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        });

      console.log("Camera permission granted.");
      console.log("Camera stream:", stream);

      streamRef.current = stream;

      recordedChunksRef.current = [];

      // =====================================================
      // MEDIA RECORDER
      // =====================================================

      let mimeType = "";

      if (
        MediaRecorder.isTypeSupported(
          "video/webm;codecs=vp8"
        )
      ) {
        mimeType = "video/webm;codecs=vp8";
      } else if (
        MediaRecorder.isTypeSupported("video/webm")
      ) {
        mimeType = "video/webm";
      }

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (
          event.data &&
          event.data.size > 0
        ) {
          recordedChunksRef.current.push(
            event.data
          );
        }
      };

      recorder.onerror = (event) => {
        console.error(
          "MediaRecorder error:",
          event
        );
      };

      recorder.onstop = async () => {
        console.log(
          "Recording stopped. Creating video..."
        );

        const chunks =
          recordedChunksRef.current;

        if (!chunks.length) {
          setPredictionError(
            "No video frames were recorded."
          );
          return;
        }

        const blob = new Blob(chunks, {
          type:
            recorder.mimeType ||
            "video/webm",
        });

        console.log(
          "Recorded video size:",
          blob.size
        );

        const recordedFile = new File(
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

      // Start recording
      recorder.start(100);

      // IMPORTANT:
      // Set camera mode AFTER stream + recorder are ready.
      // The useEffect above will then attach the stream
      // to the rendered video element.
      setVideoMode("camera");
      setCameraActive(true);

      console.log(
        "Camera recording started."
      );
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
      streamRef.current = null;
    }
  };

  // =========================================================
  // STOP CAMERA + PREDICT
  // =========================================================

  const stopCamera = () => {
    console.log(
      "Stopping camera..."
    );

    setCameraActive(false);

    // Stop recorder first.
    // onstop will create the video and send it
    // to the CNN-LSTM backend.
    if (recorderRef.current) {
      const recorder =
        recorderRef.current;

      if (
        recorder.state !== "inactive"
      ) {
        recorder.stop();
      }

      recorderRef.current = null;
    }

    // Stop camera stream
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    // Remove camera preview
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setVideoMode("none");

    console.log(
      "Camera stopped. Processing recording..."
    );
  };

  // =========================================================
  // LIVE CNN-LSTM PREDICTION
  // =========================================================

  const predictLiveVideo = async (
    videoFile
  ) => {
    try {
      setPredictionError("");
      setPrediction(null);
      setIsPredicting(true);

      const formData =
        new FormData();

      formData.append(
        "video",
        videoFile
      );

      console.log(
        "Sending recorded video to CNN-LSTM..."
      );

      const response =
        await fetch(
          "http://127.0.0.1:8000/predict-live-video",
          {
            method: "POST",
            body: formData,
          }
        );

      const data =
        await response.json();

      console.log(
        "Backend response:",
        data
      );

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Live prediction failed."
        );
      }

      setPrediction(data);

      console.log(
        "CNN-LSTM prediction:",
        data
      );
    } catch (error) {
      console.error(
        "Live prediction error:",
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
  // UPLOAD VIDEO
  // =========================================================

  const handleUploadVideo = (
    event
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) return;

    setCameraError("");
    setPredictionError("");
    setPrediction(null);

    if (
      !file.type.startsWith("video/")
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
    setUploadedVideoUrl(videoUrl);
    setVideoMode("upload");

    event.target.value = "";
  };

  // =========================================================
  // LOAD UPLOADED VIDEO
  // =========================================================

  useEffect(() => {
    if (
      videoMode === "upload" &&
      uploadedVideoUrl &&
      videoRef.current
    ) {
      const video =
        videoRef.current;

      video.pause();
      video.srcObject = null;
      video.src = uploadedVideoUrl;
      video.load();

      video
        .play()
        .catch(() => {});
    }
  }, [
    videoMode,
    uploadedVideoUrl,
  ]);

  // =========================================================
  // UPLOADED VIDEO PREDICTION
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

      const formData =
        new FormData();

      formData.append(
        "video",
        selectedFile
      );

      const response =
        await fetch(
          "http://127.0.0.1:8000/predict",
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
    if (!prediction?.label) {
      alert(
        "Please predict a sign first."
      );
      return;
    }

    if (
      !("speechSynthesis" in window)
    ) {
      alert(
        "Speech is not supported by this browser."
      );
      return;
    }

    window.speechSynthesis.cancel();

    const speech =
      new SpeechSynthesisUtterance(
        prediction.label
      );

    speech.lang = "en-US";
    speech.rate = 0.85;
    speech.pitch = 1;

    window.speechSynthesis.speak(
      speech
    );
  };

  // =========================================================
  // TEXT TO SPEECH
  // =========================================================

  const speakText = async () => {
    if (!textInput.trim()) {
      alert(
        "Please enter or speak some text first."
      );
      return;
    }

    try {
      const response =
        await fetch(
          "http://127.0.0.1:8000/tts",
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body: JSON.stringify({
              text: textInput,
              language: language,
            }),
          }
        );

      if (!response.ok) {
        const errorData =
          await response.json();

        throw new Error(
          errorData.detail ||
            "Text-to-speech failed."
        );
      }

      const audioBlob =
        await response.blob();

      const audioUrl =
        URL.createObjectURL(
          audioBlob
        );

      const audio =
        new Audio(audioUrl);

      audio.play();

      audio.onended = () => {
        URL.revokeObjectURL(
          audioUrl
        );
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
  // CONFIDENCE
  // =========================================================

  const confidenceText =
    prediction
      ? `${(
          prediction.confidence *
          100
        ).toFixed(2)}%`
      : "—";

  // =========================================================
  // SPEECH TO TEXT
  // =========================================================

  const startSpeechRecognition =
    () => {
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

      recognition.lang =
        LANGUAGE_CODES[
          language
        ];

      recognition.continuous =
        false;

      recognition.interimResults =
        false;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (
        event
      ) => {
        const text =
          event.results[0][0]
            .transcript;

        setTextInput(text);
        setIsListening(false);
      };

      recognition.onerror = (
        event
      ) => {
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

      <header className="navbar">
        <div className="logo">
          <span className="logo-icon">
            🤟
          </span>

          <span>
            ISL ASSISTANT
          </span>
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
            <br />
            Communicate effortlessly with
            real-time Indian Sign Language
            recognition and multilingual
            translation.
          </p>

        </section>

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

        {mode === "sign" && (
          <div className="content-grid">

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
                    className={
                      videoMode ===
                      "camera"
                        ? "preview-video webcam-preview"
                        : "preview-video uploaded-preview"
                    }
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

              <div className="camera-buttons">

                <button
                  className="primary-button"
                  onClick={
                    cameraActive
                      ? stopCamera
                      : startCamera
                  }
                  disabled={
                    isPredicting
                  }
                >
                  {cameraActive
                    ? "Stop & Predict"
                    : "Start Camera"}
                </button>

                <button
                  className="secondary-button"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                  disabled={
                    cameraActive ||
                    isPredicting
                  }
                >
                  Upload Video
                </button>

                <input
                  ref={
                    fileInputRef
                  }
                  type="file"
                  accept=".mp4,.webm,.mov,.avi,video/mp4,video/webm,video/quicktime,video/x-msvideo"
                  hidden
                  onChange={
                    handleUploadVideo
                  }
                />

              </div>

              {isPredicting && (
                <p className="prediction-success">
                  Processing sign with
                  CNN-LSTM...
                </p>
              )}

              {videoMode ===
                "upload" && (
                <button
                  className="primary-button predict-button"
                  onClick={
                    predictSign
                  }
                  disabled={
                    !selectedFile ||
                    isPredicting
                  }
                >
                  {isPredicting
                    ? "Predicting..."
                    : "✨ Predict Sign"}
                </button>
              )}

            </section>

            <section className="card result-card">

              <h2>
                Recognition Result
              </h2>

              <div className="result-icon">
                {prediction
                  ? "🤟"
                  : "✋"}
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
                Confidence:{" "}
                {confidenceText}
              </p>

              <select
                className="language-select"
                value={language}
                onChange={(
                  event
                ) =>
                  setLanguage(
                    event.target.value
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

            </section>

          </div>
        )}

        {mode === "text" && (
          <section className="text-sign-container card">

            <h2>
              Text to Sign
            </h2>

            <p>
              Enter text to convert it
              into Indian Sign Language.
            </p>

            <textarea
              className="text-input"
              placeholder="Enter your message here..."
              value={textInput}
              onChange={(
                event
              ) =>
                setTextInput(
                  event.target.value
                )
              }
            />

            <select
              className="language-select"
              value={language}
              onChange={(
                event
              ) =>
                setLanguage(
                  event.target.value
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
              className="secondary-button"
              onClick={
                speakText
              }
            >
              🔊 Speak Text
            </button>

            <button className="primary-button">
              Convert to Sign
            </button>

            <button
              className="secondary-button"
              onClick={
                startSpeechRecognition
              }
            >
              {isListening
                ? "🎤 Listening..."
                : "🎤 Speak"}
            </button>

          </section>
        )}

      </main>

      <footer className="footer">
        <p>
          ISL Assistant • AI-Powered
          Communication
        </p>
      </footer>

    </div>
  );
}

export default App;