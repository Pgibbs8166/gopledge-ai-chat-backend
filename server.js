const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Load your embedded content
const content = require('./content.json');

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Embed a question
async function embedText(text) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-ada-002',
    input: text
  });
  return resp.data[0].embedding;
}

// Main API route
app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'No message provided' });

    const lower = message.toLowerCase();

    // 🔒 HARDCODED OVERRIDE FOR GETTING STARTED
    if (
      lower.includes("get started") ||
      lower.includes("start fundraiser") ||
      lower.includes("launch fundraiser") ||
      (lower.includes("how do i") && lower.includes("start"))
    ) {
      return res.json({
        reply: `To get started with GoPledge, please fill out the form on our [Contact Page](https://gopledge.com/contact/). A member of our team will follow up to help you launch your fundraiser.`
      });
    }

    // Embed the user's question
    const embedding = await embedText(message);

    // Compare to existing chunks
    const sims = content.map(item => ({
      text: item.text,
      sim: cosineSimilarity(embedding, item.embedding)
    }));
    sims.sort((a, b) => b.sim - a.sim);
    const top = sims.slice(0, 3).map(i => i.text);

    // Build prompt
    const prompt = `You are an assistant trained on GoPledge's website. Use the context below to answer the question. If you don't know, say so.\n\nContext:\n${top.join('\n---\n')}\n\nQuestion: ${message}\nAnswer:`;

    // Send to GPT
    const chatResp = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.2
    });

    const reply = chatResp.choices[0].message.content;
    res.json({ reply });

  } catch (err) {
    console.error('Error in /api/chat:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Health check route
app.get('/', (req, res) => {
  res.send('GoPledge AI Chat backend is running.');
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
