import axios from "axios";
import FormData from "form-data";
import fs from "fs";

// Sends a WAV file to the speech-to-text service (stt-services/main.py) and returns the text.
export async function transcribeFile(wavPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(wavPath));

  // read here, not at import time: server.js loads .env after the imports
  const url = process.env.STT_URL || "http://localhost:8000/transcribe";
  const response = await axios.post(url, form, { headers: form.getHeaders(), timeout: 30000 });
  return response.data.text;
}
