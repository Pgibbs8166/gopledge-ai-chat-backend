const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');

const app = express();
app.use(bodyParser.json());

const content = require('./content.json');

// Init OpenAI
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

// Embed text
async function embedText(text) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-ada-002',
    input: text
  });
  return resp.data[0].embedding;
}

// Main chat route
app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'No message provided' });

    // 1. Embed the question
    const embedding = await embedText(message);

    // 2. Compare to content
    const sims = content.map(item => ({
      text: item.text,
      sim: cosineSimilarity(embedding, item.embedding)
    }));
    sims.sort((a, b) => b.sim - a.sim);
    const top = sims.slice(0, 3).map(i => i.text);

    // 3. Build prompt
    const prompt = `You are an assistant trained on GoPledge content. Use the following context to answer.\n\nContext:\n${top.join('\n---\n')}\n\nQuestion: ${message}\nAnswer:`;

    // 4. Call OpenAI Chat
    const chatResp = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are helpful and truthful.' },
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

// Health check
app.get('/', (req, res) => {
  res.send('GoPledge AI Chat backend is running.');
});

// Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
