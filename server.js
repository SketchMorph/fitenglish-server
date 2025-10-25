// server.js

import express from "express";
import cors from "cors";
import multer from "multer";
import fetch from "node-fetch";    // 👈 새로 추가
import FormData from "form-data";  // 👈 새로 추가

/* -------------------------------------------------
   기본 서버 세팅
------------------------------------------------- */
const app = express();

app.use(
  cors({
    origin: "*", // 나중에 앱 도메인만 넣어도 됨
  })
);

app.use(express.json());

/* -------------------------------------------------
   헬스체크
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
   Multer (메모리 저장)
   - 파일 디스크에 안 쓰고 req.file.buffer 로 바로 접근
------------------------------------------------- */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
});

/* -------------------------------------------------
   안전장치: 환경변수 체크
------------------------------------------------- */
if (!process.env.OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in environment!");
}

/* -------------------------------------------------
   발음 채점 유틸 함수들
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
  const dp = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0)
  );

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

  // 관사 누락
  if (/\b(a|an|the)\b/.test(ref) && !/\b(a|an|the)\b/.test(hyp)) {
    tips.push("관사(a/an/the) 발음을 분명히 해보세요.");
  }

  // 전치사 누락
  if (
    /\b(to|for|of|in|on|at)\b/.test(ref) &&
    !/\b(to|for|of|in|on|at)\b/.test(hyp)
  ) {
    tips.push("전치사(to/for/of 등)를 빠뜨리지 않도록 해보세요.");
  }

  // 억양 피드백
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
  const accuracy = Math.max(
    0,
    Math.round((1 - dist / maxLen) * 100)
  );

  return { accuracy, tips: buildTips(ref, hyp) };
}

/* -------------------------------------------------
   Whisper 호출 (재시도 포함)
   - OpenAI SDK 없이 우리가 직접 multipart/form-data를 만든다
   - ECONNRESET 줄이기 위한 전략
------------------------------------------------- */
async function transcribeWithRetryMem(audioBuffer, filename, tries = 3) {
  let lastErr;

  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      console.log(`[transcribeWithRetryMem] attempt ${attempt} ...`);

      // 🔸 multipart/form-data 생성
      const fd = new FormData();

      // file 파트
      fd.append("file", audioBuffer, {
        filename: filename || "speech.m4a",
        contentType: "audio/m4a",
      });

      // model 파트 (OpenAI STT 모델)
      fd.append("model", "gpt-4o-mini-transcribe");

      // 필요하다면 언어 힌트 (영어만 할 거면 넣어도 됨)
      // fd.append("language", "en");

      // 🔸 fetch 로 직접 호출
      const resp = await fetch(
        "https://api.openai.com/v1/audio/transcriptions",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
            ...fd.getHeaders(), // form-data가 boundary 포함한 Content-Type 생성
          },
          body: fd,
        }
      );

      if (!resp.ok) {
        const errText = await resp.text();
        console.error(
          "[transcribeWithRetryMem] non-200 from OpenAI:",
          resp.status,
          errText
        );
        throw new Error(
          `OpenAI STT failed ${resp.status}: ${errText}`
        );
      }

      const json = await resp.json();

      if (!json.text) {
        throw new Error("No text in transcription response");
      }

      // 성공적으로 텍스트 받음
      return json.text;
    } catch (err) {
      console.error(
        "[transcribeWithRetryMem] failed:",
        err?.status ||
          err?.code ||
          err?.message ||
          err
      );
      lastErr = err;

      // 짧게 쉰 뒤 재시도
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  // 전부 실패하면 마지막 에러 던짐
  throw lastErr;
}

/* -------------------------------------------------
   /speech/score
   - 프런트에서 FileSystem.uploadAsync 로 보내는 엔드포인트
   - fieldName: "audio"
   - parameters: { target: "..." }
   - 응답: { ok, transcript, accuracy, tips }
------------------------------------------------- */
app.post("/speech/score", upload.single("audio"), async (req, res) => {
  try {
    // 1) 유효성 확인
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "audio field is required" });
    }
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "Server missing OPENAI_API_KEY",
      });
    }

    const targetSentence = String(req.body?.target || "");

    // 2) Whisper 호출 (음성 -> 텍스트)
    const transcript = await transcribeWithRetryMem(
      req.file.buffer,
      req.file.originalname || "speech.m4a",
      3
    );

    // 3) 채점 계산
    const { accuracy, tips } = simpleTextScore(
      targetSentence,
      transcript
    );

    // 4) 결과 응답
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
   /transcribe (디버그용: 파일 -> 텍스트만)
   - post field: "file"
   - 응답: { text }
------------------------------------------------- */
app.post("/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "file field is required" });
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
   서버 시작
------------------------------------------------- */
const PORT = Number(process.env.PORT || 4000);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
