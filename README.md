# 🤖 AI Chatbot using Chainlit 🚀

A modern AI-powered chatbot built with **Python**, **Chainlit**, and **OpenAI (Gemini)**.  
This bot can interact intelligently with users and is designed for easy deployment and scalability.

---

## 📂 **Project Structure**

AI_CHATBOT_USING_CHAINLIT/
├── .chainlit/ # Chainlit configuration
├── .venv/ # Python virtual environment (ignored)
├── .env # Environment variables (ignored)
├── main.py # Main chatbot script
├── chainlit.md # Chainlit config doc (optional)
├── .gitignore # Git ignore rules
├── pyproject.toml # Python project config
├── README.md # Project README (this file!)
└── uv.lock # Lock file for dependencies


---

## ⚙️ **Setup & Run**

### 1️⃣ Clone the repo

```bash
git clone https://github.com/SaadALam7865/Custom_ChatGPT_with_Python_Chainlit-
cd AI_CHATBOT_USING_CHAINLIT

# Create virtual environment (Linux/macOS)
python3 -m venv .venv

# Or for Windows
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

pip install -r requirements.txt

pip install .

GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

chainlit run main.py

🌐 Deploy for Free
Easily deploy on Railway or Render:

✅ Push your code to GitHub
✅ Add your GEMINI_API_KEY in the platform’s environment variables
✅ Use chainlit run main.py as the start command
✅ Get a free public link for your chatbot!

🛡️ Best Practices
✅ .env and .venv are ignored — safe from Git leaks
✅ Use secrets manager on deployment
✅ Keep API keys private!

💡 Technologies Used
🐍 Python

🔗 Chainlit

🤖 OpenAI Gemini API

☁️ .env for secrets

👑 Author
Developed by: [render or railway or https://github.com/SaadALam7865]
Feel free to ⭐️ this repo & connect!

📜 License
This project is open-source — free to use, learn, and modify!

Happy Building! 🚀✨