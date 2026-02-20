import dotenv from "dotenv";
dotenv.config();

console.log("API KEY LOADED:", process.env.OPENAI_API_KEY?.slice(0, 8));


import express, { Request, Response, NextFunction } from "express";
import OpenAI from "openai";

// --------------------
// Types
// --------------------

interface ChatRequestBody {
  message: string;
}

// --------------------
// OpenAI Client
// --------------------

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || ""
});

//RAG Setup:
async function embedText(text: string): Promise<number[]> {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text
  });

  return embedding.data[0].embedding;
}

type Document = {
  content: string;
  embedding: number[];
};

const documents: Document[] = [];

async function seedDocuments() {
  const texts = [
    "REST APIs use HTTP methods and are stateless.",
    "AWS Lambda is a serverless compute service.",
    "Express.js is a minimal Node.js web framework."
  ];

  for (const text of texts) {
    documents.push({
      content: text,
      embedding: await embedText(text)
    });
  }
}

seedDocuments();

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }

  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function retrieveContext(query: string): Promise<string> {
  const queryEmbedding = await embedText(query);

  const scored = documents.map(doc => ({
    content: doc.content,
    score: cosineSimilarity(queryEmbedding, doc.embedding)
  }));

  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, 2).map(d => d.content).join("\n");
}

// --------------------
// Express App Setup
// --------------------

const app = express();
const PORT = 3000;

// Built-in JSON body parser
app.use(express.json());

// --------------------
// Validation Middleware
// --------------------

function validateChatInput(
  req: Request<{}, {}, ChatRequestBody>,
  res: Response,
  next: NextFunction
) {
  const { message } = req.body;

  if (!message) {
    return res.status(400).json({
      error: "Message field is required"
    });
  }

  if (typeof message !== "string") {
    return res.status(400).json({
      error: "Message must be a string"
    });
  }

  if (message.trim().length === 0) {
    return res.status(400).json({
      error: "Message cannot be empty"
    });
  }

  next();
}

app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.url} - ${duration}ms`);
  });

  next();
});

// --------------------
// Routes
// --------------------

// Health Check
app.get("/health", (req: Request, res: Response) => {
  res.status(200).json({
    status: "OK",
    uptime: process.uptime()
  });
});

app.post("/chat", validateChatInput, async (req: Request, res: Response) => {
  try {
    const { message } = req.body;

    const context = await retrieveContext(message);

    const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    max_tokens: 150,
    temperature: 0.7,
    messages: [
      {
        role: "system",
        content: "Answer using the context below.\n\n" + context
      },
      { role: "user", content: message }
  ]
});

app.post("/chat-stream", validateChatInput, async (req: Request, res: Response) => {
  try {
    const { message } = req.body;

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const stream = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      stream: true,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: message }
      ]
    });

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content;

      if (token) {
        res.write(token);
      }
    }

    res.end();

  } catch (err) {
    console.error(err);
    res.end();
  }
});

    const reply = completion.choices[0].message.content;

    // logging for cost + token awareness
    console.log("Tokens used:", completion.usage);

    res.json({
      reply
    });

  } catch (error: any) {
    console.error("OpenAI Error:", error);

    res.status(500).json({
      error: "AI service failed"
    });
  }
});


// --------------------
// Start Server
// --------------------

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
