"""
Fast Test Flask Server - No Model Loading
For quick React frontend testing
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Dummy questions for testing
DUMMY_QUESTIONS = [
    "這個疾病的症狀有哪些？",
    "需要注意什麼飲食禁忌？",
    "藥物的副作用是什麼？",
    "如何進行日常照護？",
    "什麼時候需要就醫？",
    "懷孕期間可以吃什麼食物？",
    "如何控制血糖？",
    "運動對病情有幫助嗎？",
    "需要定期檢查什麼項目？",
    "如何預防併發症？"
]

# Dummy answers for different specialties
DUMMY_ANSWERS = {
    "gdm": "這是關於妊娠期糖尿病的回答。建議您保持健康飲食，定期監測血糖，並遵循醫生的建議。",
    "ckd": "這是關於慢性腎臟病的回答。請注意控制蛋白質攝取，避免高磷食物，並定期檢查腎功能。",
    "ppd": "這是關於產後憂鬱症的回答。請多休息，尋求家人支持，必要時諮詢心理醫生。如有嚴重情緒問題，請立即就醫。"
}

@app.route("/")
def index():
    return jsonify({
        "message": "Test Flask Server is Running!",
        "status": "OK",
        "note": "This is a lightweight test server without model loading"
    })

@app.route("/api/chat", methods=["GET"])
def api_chat():
    """API endpoint for React frontend to get initial data"""
    print("📥 GET /api/chat - Fetching questions...")
    
    random_question = random.choice(DUMMY_QUESTIONS)
    
    response = {
        "questions": DUMMY_QUESTIONS,
        "random_question": random_question,
        "success": True
    }
    
    print(f"✅ Returning {len(DUMMY_QUESTIONS)} questions")
    return jsonify(response)

@app.route("/upload", methods=["POST"])
def upload_audio():
    """Dummy audio upload endpoint"""
    print("📥 POST /upload - Audio upload (dummy)")
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Return dummy transcription
    dummy_transcription = "這是測試語音辨識的結果"
    
    print(f"✅ Returning dummy transcription: {dummy_transcription}")
    return dummy_transcription

@app.route("/ask", methods=["POST"])
def ask():
    """Dummy chat endpoint"""
    question = request.form.get("question", "")
    role = request.form.get('role', 'unknown')
    model_type = request.form.get('model_type', 'gdm')
    responseWithAudio = request.form.get("responseWithAudio", "false")
    
    print(f"📥 POST /ask")
    print(f"   Question: {question}")
    print(f"   Role: {role}")
    print(f"   Model: {model_type}")
    print(f"   With Audio: {responseWithAudio}")
    
    if not question:
        return jsonify({"error": "請輸入問題"}), 400
    
    # Simulate processing time
    time.sleep(1)
    
    # Get dummy answer based on model type
    base_answer = DUMMY_ANSWERS.get(model_type, DUMMY_ANSWERS["gdm"])
    answer = f"針對您的問題「{question}」，{base_answer}"
    
    print(f"✅ Returning dummy answer (length: {len(answer)})")
    
    # Don't generate audio in test mode
    if responseWithAudio == "true":
        print("⚠️  Audio requested but skipped in test mode")
        return jsonify({
            "answer": answer,
            "note": "Audio generation skipped in test mode"
        })
    else:
        return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "server": "test",
        "timestamp": time.time()
    })

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Test Flask Server (No Models)")
    print("=" * 60)
    print("✅ CORS enabled")
    print("✅ Dummy data ready")
    print("📡 Server will run on: http://0.0.0.0:5012")
    print("=" * 60)
    print("\n🔗 Available endpoints:")
    print("   GET  /           - Server info")
    print("   GET  /api/chat   - Get questions list")
    print("   POST /upload     - Dummy audio upload")
    print("   POST /ask        - Dummy Q&A")
    print("   GET  /health     - Health check")
    print("\n💡 This is a fast test server for React development")
    print("   No models loaded, instant responses!\n")
    
    app.run(host="0.0.0.0", port=5012, debug=True)
