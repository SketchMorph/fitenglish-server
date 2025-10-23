import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const app = express();

// ===== Middleware =====
app.use(cors());
app.use(express.json());

// ✅ 루트 경로 (Render 헬스체크 대응)
app.get("/", (req, res) => {
  res.status(200).send("ok");
});

// ✅ 헬스체크 (Render 모니터링용)
app.get("/healthz", (req, res) => {
  res.status(200).send("ok");
});
app.get("/health", (_req, res) => res.json({ ok: true, time: Date.now() }));

// ===== 업로드 임시 폴더 =====
const UPLOADS = path.join(process.cwd(), "uploads");
if (!fs.existsSync(UPLOADS)) fs.mkdirSync(UPLOADS, { recursive: true });
const upload = multer({ dest: UPLOADS });

// ===== OpenAI 클라이언트 =====
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * 1) /transcribe
 *   - 클라이언트 필드명: "file"
 *   - 리턴: { text }
 */
app.post("/transcribe", upload.single("file"), async (req, res) => {
  let tempPath;
  try {
    if (!req.file) return res.status(400).json({ error: "file field is required" });
    tempPath = req.file.path;

    const resp = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempPath),
      model: "whisper-1",
      language: "en",
    });

    res.json({ text: resp.text || "" });
  } catch (err) {
    console.error("[/transcribe] error:", err);
    res.status(500).json({ error: err.message || "transcribe failed" });
  } finally {
    if (tempPath) {
      try {
        fs.unlinkSync(tempPath);
      } catch {}
    }
  }
});

/**
 * 2) /speech/score
 *   - 클라이언트 필드명: "audio", "target"
 *   - 리턴: { transcript, accuracy, tips: [] }
 */
app.post("/speech/score", upload.single("audio"), async (req, res) => {
  let tempPath;
  try {
    const target = String(req.body?.target || "");
    if (!req.file) return res.status(400).json({ error: "audio field is required" });
    tempPath = req.file.path;

    // 2-1. Whisper로 음성 → 텍스트 변환
    const tr = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempPath),
      model: "whisper-1",
      language: "en",
    });
    const transcript = tr.text || "";

    // 2-2. 간단 채점
    const { accuracy, tips } = simpleTextScore(target, transcript);

    res.json({ transcript, accuracy, tips });
  } catch (err) {
    console.error("[/speech/score] error:", err);
    res.status(500).json({ error: err.message || "score failed" });
  } finally {
    if (tempPath) {
      try {
        fs.unlinkSync(tempPath);
      } catch {}
    }
  }
});

/* ======= 채점 유틸 ======= */
function normalize(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[^a-z0-9' ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function levenshtein(a, b) {
  const m = a.length,
    n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
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
  if (hyp.length < ref.length * 0.7)
    tips.push("문장을 끝까지 또박또박 읽어보세요.");
  if (/\b(a|an|the)\b/.test(ref) && !/\b(a|an|the)\b/.test(hyp))
    tips.push("관사(a/an/the) 발음을 분명히 해보세요.");
  if (/\b(to|for|of|in|on|at)\b/.test(ref) && !/\b(to|for|of|in|on|at)\b/.test(hyp))
    tips.push("전치사(to/for/of 등)를 빠뜨리지 않도록 해보세요.");
  if (tips.length === 0)
    tips.push("자연스러운 강세와 끊어 읽기를 연습해보세요.");
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

// ===== Start Server =====
const PORT = Number(process.env.PORT || 4000);
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});

// server/server.js (기존 코드 하단에 추가)
app.post("/speech/auto", upload.single("audio"), async (req, res) => {
  let tempPath;
  try {
    const target = String(req.body?.target || "");
    if (!req.file) return res.status(400).json({ error: "audio field is required" });
    tempPath = req.file.path;

    // Whisper 전사
    const tr = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempPath),
      model: "whisper-1",
      language: "en",
    });
    const transcript = tr.text || "";

    // 채점
    const { accuracy, tips } = simpleTextScore(target, transcript);
    res.json({ transcript, accuracy, tips });
  } catch (err) {
    console.error("[/speech/auto] error:", err);
    res.status(500).json({ error: err.message || "speech/auto failed" });
  } finally {
    if (tempPath) {
      try { fs.unlinkSync(tempPath); } catch {}
    }
  }
});

