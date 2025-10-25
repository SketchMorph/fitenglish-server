// server/server.js

import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

// --- Node 환경 polyfill: File이 없을 수 있으므로 만들어줌
if (typeof File === "undefined") {
  global.File = class NodeFile extends Blob {
    constructor(parts, name, options = {}) {
      super(parts, options);
      this.name = name;
      this.lastModified = options?.lastModified || Date.now();
    }
  };
}

const app = express();

app.use(
  cors({
    origin: "*",
  })
);
app.use(express.json());

// 헬스체크
app.get("/", (_req, res) => res.status(200).send("ok"));
app.get("/healthz", (_req, res) => res.status(200).send("ok"));
app.get("/health", (_req, res) =>
  res.json({ ok: true, time: Date.now() })
);

// Multer: 디스크 안 쓰고 메모리에 저장
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

// OpenAI 클라이언트
if (!process.env.OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY in environment!");
}
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ===== 발음 채점 유틸 =====
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
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

function buildTips(ref, hyp) {
  const tips = [];
  if (hyp.length < ref.length * 0.7) {
    tips.push("문장을 끝까지 또박또박 읽어보세요.");
  }
  if (/\b(a|an|the)\b/.test(ref) && !/\b(a|an|the)\b/.test(hyp)) {
    tips.push("관사(a/an/the) 발음을 분명히 해보세요.");
  }
  if (
    /\b(to|for|of|in|on|at)\b/.test(ref) &&
    !/\b(to|for|of|in|on|at)\b/.test(hyp)
  ) {
    tips.push("전치사(to/for/of 등)를 빠뜨리지 않도록 해보세요.");
  }
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

// ===== Whisper(STT) 호출 (재시도 포함) =====
async function transcribeWithRetryMem(audioBuffer, filename, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      console.log(`[transcribeWithRetryMem] attempt ${i + 1} ...`);

      // Buffer -> Blob -> File (polyfill File ensures Node compatibility)
      const blob = new Blob([audioBuffer], { type: "audio/m4a" });
      const fileObj = new File(
        [blob],
        filename || "speech.m4a",
        { type: "audio/m4a" }
      );

      // OpenAI STT (최신 STT 모델 사용)
      const resp = await openai.audio.transcriptions.create({
        file: fileObj,
        model: "gpt-4o-mini-transcribe",
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
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  throw lastErr;
}

// ===== /speech/score =====
// - 프런트가 fieldName: "audio" 로 올리는 걸 받음
// - parameters: { target: "..."} 도 여기서 읽힘
app.post("/speech/score", upload.single("audio"), async (req, res) => {
  try {
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

    // 1) 음성 변환
    const transcript = await transcribeWithRetryMem(
      req.file.buffer,
      req.file.originalname || "speech.m4a",
      3
    );

    // 2) 채점
    const { accuracy, tips } = simpleTextScore(
      targetSentence,
      transcript
    );

    // 3) 응답
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

// ===== /transcribe (디버그용: 단순 음성->텍스트만 뽑기)
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

// 서버 시작
const PORT = Number(process.env.PORT || 4000);
app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
