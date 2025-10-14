const express = require('express');
const bodyParser = require('body-parser');
const { Configuration, OpenAIApi } = require('openai');

const app = express();
app.use(bodyParser.json());

const content = require('./content.json');

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
});
const openai = new OpenAIApi(configuration);

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function embedText(text) {
  const resp = await openai.createEmbedding({
    model: 'text-embedding-ada-002',
    input: text
  });
  return resp.data.data[0].embedding;
}

app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'No message provided' });

    const embedding = await embedText(message);
    const sims = content.map(item => ({
      text: item.text,
      sim: cosineSimilarity(embedding, item.embedding)
    }));
    sims.sort((a,b) => b.sim - a.sim);
    const top = sims.slice(0, 3).map(i => i.text);

    const prompt = `You are an assistant trained on GoPledge content. Use the following context to answer.\n\nContext:\n${top.join('\n---\n')}\n\nQuestion: ${message}\nAnswer:`;

    const chatResp = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are helpful and truthful.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.2
    });

    const reply = chatResp.data.choices[0].message.content;
    res.json({ reply });
  } catch (err) {
    console.error('Error in /api/chat:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/', (req, res) => {
  res.send('GoPledge AI Chat backend is running.');
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
