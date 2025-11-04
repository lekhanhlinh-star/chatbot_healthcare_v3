print("Import library")
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import whisper
import torch
# from TTS.api import TTS

# Import shared models first to initialize them
print("Loading shared models...")
import shared_models

# Import RAG inference modules (will use shared models)
from rag_inference_gdm import qa_chain as qa_chain_gdm, compression_retriever as compression_retriever_gdm, related_question_chain as related_question_chain_gdm
from rag_inference_ckd import qa_chain as qa_chain_ckd, compression_retriever as compression_retriever_ckd, related_question_chain as related_question_chain_ckd
from rag_inference_ppd import qa_chain as qa_chain_ppd, compression_retriever as compression_retriever_ppd, related_question_chain as related_question_chain_ppd

from opencc import OpenCC
import uuid
import edge_tts
import base64
import random
import time

print("Finish import")
myuuid = uuid.uuid4()
app = Flask(__name__)
UPLOAD_FOLDER = "temp"
AUDIO_CLONE = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#init model here
print("init model")
asr_model = whisper.load_model("small")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
cc = OpenCC('s2twp')
print("Finish init model")
def llm_inference_gdm(user_query):
    """GDM (妊娠期糖尿病) inference"""
    question = user_query
    answer = qa_chain_gdm.invoke(question)
    return answer

def llm_inference_ckd(user_query):
    """CKD (慢性腎臟病) inference"""
    question = user_query
    answer = qa_chain_ckd.invoke(question)
    return answer

def llm_inference_ppd(user_query):
    """PPD (產後憂鬱症) inference"""
    question = user_query
    answer = qa_chain_ppd.invoke(question)
    return answer

def llm_inference(user_query, model_type="gdm"):
    """通用inference函數，根據model_type選擇對應的模型"""
    if model_type == "ckd":
        return llm_inference_ckd(user_query)
    elif model_type == "ppd":
        return llm_inference_ppd(user_query)
    else:  # 預設使用 gdm
        return llm_inference_gdm(user_query)

def load_questions(model_type: str = "gdm"):
    """Load questions file based on model_type/role.

    Supported model_type values: 'gdm', 'ckd', 'ppd'. Defaults to 'gdm'.
    Falls back to a small set of default questions when file is missing.
    """
    mapping = {
        "gdm": "gdm_questions.txt",
        "ckd": "ckd_questions.txt",
        "ppd": "ppd_questions.txt",
    }

    filename = mapping.get(model_type, mapping["gdm"])  # default to gdm
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions
    except FileNotFoundError:
        print(f"⚠️  {filename} not found, using default questions")
        return [
            "這個疾病的症狀有哪些？",
            "需要注意什麼飲食禁忌？",
            "藥物的副作用是什麼？",
        ]

@app.route("/")
def index():
    """Trang chọn chuyên khoa"""
    return render_template("index.html")

@app.route("/chat")
def chat():
    """Trang chat với chatbot - HTML template version"""
    # Allow selecting model_type via query param, e.g. /chat?model_type=ckd
    model_type = request.args.get('model_type', 'gdm')
    questions = load_questions(model_type)
    random_question = random.choice(questions) if questions else ""
    return render_template("chat.html", questions=questions, random_question=random_question)

@app.route("/api/chat", methods=["GET"])
def api_chat():
    """API endpoint for React frontend to get initial data"""
    print("📥 GET /api/chat - Fetching questions...")
    
    try:
        # allow the frontend to request questions for a specific model_type
        model_type = request.args.get('model_type', 'gdm')
        questions = load_questions(model_type)
        random_question = random.choice(questions) if questions else ""
        
        response = {
            "questions": questions,
            "random_question": random_question,
            "success": True
        }

        print(f"✅ Returning {len(questions)} questions for model_type={model_type}")
        return jsonify(response)
    except Exception as e:
        print(f"❌ Error in /api/chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "questions": [],
            "random_question": ""
        }), 500

@app.route("/upload", methods=["POST"])
def upload_audio():
    print("📥 POST /upload - Audio upload")
    
    if "audio" not in request.files:
        print("❌ No audio file in request")
        return "沒有音訊檔案", 400

    file = request.files["audio"]
    
    if file.filename == "":
        print("❌ Empty filename")
        return "檔案名稱為空", 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"✅ Audio saved: {filepath}")

        print("🎤 Transcribing with Whisper...")
        result = asr_model.transcribe(filepath, language="zh")
        
        # Bạn có thể gọi ASR để chuyển đổi giọng nói thành văn bản ở đây
        # ví dụ: result = my_asr(filepath)
        answer = cc.convert(result['text'])
        
        print(f"✅ Transcription: {answer}")
        return answer
    except Exception as e:
        print(f"❌ Error in upload: {str(e)}")
        return f"錯誤: {str(e)}", 500




@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.route("/ask", methods=["POST"])
async def ask():
    print("📥 POST /ask - Processing question")
    
    try:
        question = request.form.get("question")
        role = request.form.get('role', 'unknown')
        gender = request.form.get('gender', 'female')
        model_type = request.form.get('model_type', 'gdm')  # 預設使用 gdm
        responseWithAudio = request.form.get("responseWithAudio", False)
        
        print(f"   Question: {question}")
        print(f"   Role: {role}")
        print(f"   Model: {model_type}")
        print(f"   Audio: {responseWithAudio}")
        
        if not question:
            print("❌ No question provided")
            return "請輸入問題", 400

        start_time = time.time()
        print(f"🔄 Running LLM inference with {model_type} model...")
        answer = llm_inference(question, model_type)
        end_time = time.time() - start_time
        print(f"✅ LLM: {end_time:.2f}s")
        
        # Remove <think>...</think> tags server-side before any further processing
        import re
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
        
        start_time = time.time()
        answer = cc.convert(answer)
        end_time = time.time() - start_time
        print(f"✅ Convert: {end_time:.2f}s")
        
        start_time = time.time()
        if responseWithAudio == "true":
            print("🎵 Generating audio...")
            # select voice by gender (male -> Yunyang, female -> Xiaoxiao)
            if str(gender).lower() == 'male':
                voices = "zh-CN-YunyangNeural"
            else:
                voices = "zh-CN-XiaoxiaoNeural"
            myuuid = uuid.uuid4()
            audio_name = str(myuuid) + '.mp3'
            filepath = os.path.join(UPLOAD_FOLDER, audio_name)
            tts = edge_tts.Communicate(
                    text=answer,
                    voice=voices)
            await tts.save(filepath)
            # return jsonify({"answer": answer, "audio": audio, "filepath": filepath})
            with open(filepath, 'rb') as audio_file:
                audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            end_time = time.time() - start_time
            print(f"✅ Audio: {end_time:.2f}s")
            return jsonify({"answer": answer, "audio_base64": audio_base64})
        else:
            print("✅ Returning text answer")
            return jsonify({"answer": answer})
            
    except Exception as e:
        print(f"❌ Error in ask: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"處理問題時發生錯誤: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
