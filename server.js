// server.js

import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

/* -------------------------------------------------
   Node 환경에서 File이 없는 경우를 위한 polyfill
   - OpenAI SDK는 브라우저식 File/Blob도 지원한다고 가정하므로
   - Render(Node)에서도 같은 인터페이스를 흉내내준다.
------------------------------------------------- */
if (typeof File === "undefined") {
  global.File = class NodeFile extends Blob {
    constructor(parts, name, options = {}) {
      super(parts, options);
      this.name = name;
      this.lastModified = options?.lastModified || Date.now();
    }
  };
}

/* -------------------------------------------------
   기본 서버 세팅
------------------------------------------------- */
const app = express();

// CORS (지금은 전체 허용, 나중에 앱 도메인만 허용해도 됨)
app.use(
  cors({
    origin: "*",
  })
);

// JSON 바디 파싱
app.use(express.json());

/* -------------------------------------------------
   헬스체크 라우트
   - Render 상태 확인, 모바일 앱에서 ping 등
------------------------------------------------- */
app.get("/", (_req, res) => {
  res.status(200).send("ok");
});

app.get("/healthz", (_req, res) => {
  res.status(200).send("ok");
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, time: Date.now() });
});

/* -------------------------------------------------
   Multer 설정 (메모리 저장)
   - 파일을 디스크에 쓰지 않고 req.file.buffer로 바로 사용
------------------------------------------------- */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB 제한
  },
});

/* -------------------------------------------------
   OpenAI 클라이언트
   - 키는 Render 대시보드 Environment Vars 에만 넣기
   - 로컬 .env나 깃에 절대 커밋하지 말 것
------------------------------------------------- */
if (!process.env.OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in environment!");
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/* -------------------------------------------------
   발음/문장 채점 유틸 함수들
   - 사용자가 읽어야 할 문장(target) vs 실제 Whisper로 인식된 문장(transcript)
   - accuracy: 0~100
   - tips: 최대 3개 간단 피드백
------------------------------------------------- */
function normalize(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[^a-z0-9' ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1, // 삭제
        dp[i][j - 1] + 1, // 삽입
        dp[i - 1][j - 1] + cost // 치환
      );
    }
  }

  return dp[m][n];
}

function buildTips(ref, hyp) {
  const tips = [];

  // 너무 짧게 읽었을 때
  if (hyp.length < ref.length * 0.7) {
    tips.push("문장을 끝까지 또박또박 읽어보세요.");
  }

  // 관사 누락 체크
  if (/\b(a|an|the)\b/.test(ref) && !/\b(a|an|the)\b/.test(hyp)) {
    tips.push("관사(a/an/the) 발음을 분명히 해보세요.");
  }

  // 전치사 누락 체크
  if (
    /\b(to|for|of|in|on|at)\b/.test(ref) &&
    !/\b(to|for|of|in|on|at)\b/.test(hyp)
  ) {
    tips.push("전치사(to/for/of 등)를 빠뜨리지 않도록 해보세요.");
  }

  // 위에 안 걸리면 억양/리듬 피드백
  if (tips.length === 0) {
    tips.push("자연스러운 강세와 끊어 읽기를 연습해보세요.");
  }

  return tips.slice(0, 3);
}

function simpleTextScore(reference, hypothesis) {
  const ref = normalize(reference);
  const hyp = normalize(hypothesis || "");

  const dist = levenshtein(ref, hyp);
  const maxLen = Math.max(ref.length, hyp.length) || 1;
  const accuracy = Math.max(0, Math.round((1 - dist / maxLen) * 100));

  return { accuracy, tips: buildTips(ref, hyp) };
}

/* -------------------------------------------------
   Whisper(STT) 호출 함수 (재시도 포함)
   - React Native -> 서버로 올라온 녹음파일(버퍼) -> Blob -> File
   - OpenAI audio.transcriptions.create 로 전송
   - OpenAI에서 "multipart form 파싱 불가" 400 에러가 났던 부분 해결
------------------------------------------------- */
async function transcribeWithRetryMem(audioBuffer, filename, tries = 3) {
  let lastErr;

  for (let i = 0; i < tries; i++) {
    try {
      console.log(`[transcribeWithRetryMem] attempt ${i + 1} ...`);

      // Buffer -> Blob
      const blob = new Blob([audioBuffer], { type: "audio/m4a" });

      // Blob -> File (polyfill된 File 사용)
      const fileObj = new File(
        [blob],
        filename || "speech.m4a",
        { type: "audio/m4a" }
      );

      // OpenAI Whisper / STT
      const resp = await openai.audio.transcriptions.create({
        file: fileObj,
        model: "gpt-4o-mini-transcribe",
        // language: "en", // 필요하면 강제 지정
      });

      if (!resp || !resp.text) {
        throw new Error("No text in transcription response");
      }

      return resp.text;
    } catch (err) {
      console.error(
        "[transcribeWithRetryMem] failed:",
        err?.status || err?.code || err?.message || err
      );

      lastErr = err;
      // 약간 쉬고 재시도
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  throw lastErr;
}

/* -------------------------------------------------
   /transcribe  (디버그/테스트용)
   - form-data field: "file"
   - 결과: { text }
   - 실제 앱에 안 써도 되지만 STT 단독 테스트용으로 유용
------------------------------------------------- */
app.post("/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "file field is required" });
    }

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "Server missing OPENAI_API_KEY",
      });
    }

    const text = await transcribeWithRetryMem(
      req.file.buffer,
      req.file.originalname || "audio.m4a",
      3
    );

    return res.json({ text });
  } catch (err) {
    console.error("[/transcribe] error:", err);
    return res.status(500).json({
      error: {
        message: err.message || "transcribe failed",
        code: err.code,
        status: err.status,
      },
    });
  }
});

/* -------------------------------------------------
   /speech/score
   - form-data fields:
        audio   -> 녹음 파일
        target  -> 사용자가 읽어야 할 원문 문장
   - 응답:
        {
          ok: true,
          transcript: "...",
          accuracy: 87,
          tips: ["...", "..."]
        }
------------------------------------------------- */
app.post("/speech/score", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "audio field is required" });
    }

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "Server missing OPENAI_API_KEY",
      });
    }

    const targetSentence = String(req.body?.target || "");

    // 1) 음성 -> 텍스트
    const transcript = await transcribeWithRetryMem(
      req.file.buffer,
      req.file.originalname || "speech.m4a",
      3
    );

    // 2) 발음 채점
    const { accuracy, tips } = simpleTextScore(targetSentence, transcript);

    // 3) 결과 반환
    return res.json({
      ok: true,
      transcript,
      accuracy,
      tips,
    });
  } catch (err) {
    console.error("[/speech/score] error:", err);

    return res.status(500).json({
      ok: false,
      error: {
        message: err.message || "Connection error.",
        code: err.code,
        status: err.status,
      },
    });
  }
});

/* -------------------------------------------------
   서버 시작
   - Render에서는 PORT env를 자동으로 준다 (우리는 10000으로 세팅)
   - 로컬에서는 기본 4000
------------------------------------------------- */
const PORT = Number(process.env.PORT || 4000);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
