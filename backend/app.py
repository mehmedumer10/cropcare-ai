"""
CropCare AI - Backend API Server
Run with: python app.py
Requires: pip install flask flask-cors tensorflow pillow numpy joblib
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import joblib
import json
import io
import sys
import base64
import traceback

app = Flask(__name__)
CORS(app)  # Allow Android app to connect

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR    = os.environ.get("CROPCARE_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")
IMG_SIZE     = (128, 128)

# ─── Treatment database (same as original) ───────────────────────────────
TREATMENTS = {
    "Bacterialblight": {
        "description": "Bacterial Blight is caused by Xanthomonas bacteria. Leaves show water-soaked lesions that turn yellow then brown.",
        "severity": "High",
        "treatment": ["Apply copper-based bactericides", "Remove infected leaves immediately", "Avoid overhead irrigation", "Use disease-free seeds"],
        "fertilizer": "Reduce nitrogen fertilizer. Apply potassium to strengthen plant immunity.",
        "prevention": "Use resistant varieties. Maintain field hygiene. Rotate crops annually."
    },
    "Banded_Chlorosis": {
        "description": "Banded Chlorosis appears as yellow bands across leaves, caused by nutrient deficiency or viral infection.",
        "severity": "Medium",
        "treatment": ["Apply zinc and iron micronutrients", "Check soil pH (ideal 6.0-7.0)", "Foliar spray with zinc sulfate"],
        "fertilizer": "Apply NPK 20-20-20 with micronutrients. Zinc sulfate foliar spray 0.5%.",
        "prevention": "Maintain proper soil pH. Regular soil testing. Balanced fertilization."
    },
    "BlackPoint": {
        "description": "Black Point is a fungal disease affecting wheat grains, causing dark discoloration at the embryo end.",
        "severity": "Medium",
        "treatment": ["Apply fungicides containing tebuconazole or propiconazole", "Harvest at proper maturity", "Dry grain quickly after harvest"],
        "fertilizer": "Balanced NPK. Avoid excess nitrogen which increases susceptibility.",
        "prevention": "Use certified clean seed. Avoid late planting. Ensure good air circulation."
    },
    "Blast": {
        "description": "Rice/Wheat Blast is a devastating fungal disease. Grey-green lesions appear on leaves, spreading rapidly.",
        "severity": "Very High",
        "treatment": ["Apply tricyclazole or isoprothiolane immediately", "Emergency fungicide spray every 7 days", "Remove heavily infected plants"],
        "fertilizer": "Reduce nitrogen. Apply silicon-based fertilizers to strengthen cell walls.",
        "prevention": "Plant resistant varieties. Balanced nitrogen. Avoid dense planting."
    },
    "Blight": {
        "description": "Blight causes rapid browning and death of plant tissue. Multiple pathogens can cause blight.",
        "severity": "High",
        "treatment": ["Apply broad-spectrum fungicide", "Remove all infected plant material", "Improve drainage to reduce moisture"],
        "fertilizer": "Apply potassium phosphite to boost plant immunity.",
        "prevention": "Crop rotation. Avoid waterlogging. Use resistant varieties."
    },
    "BrownRust": {
        "description": "Brown Rust appears as small orange-brown pustules on leaves. Spreads rapidly in warm humid conditions.",
        "severity": "High",
        "treatment": ["Apply propiconazole or tebuconazole fungicide", "Spray at first sign of rust", "Repeat after 14 days"],
        "fertilizer": "Reduce excess nitrogen. Ensure adequate potassium for resistance.",
        "prevention": "Plant rust-resistant wheat varieties. Early planting to avoid peak rust season."
    },
    "Brown_Spot": {
        "description": "Brown Spot (Helminthosporium) causes oval brown lesions with yellow halos on rice/corn leaves.",
        "severity": "Medium",
        "treatment": ["Apply mancozeb or iprodione fungicide", "Improve soil fertility", "Ensure proper drainage"],
        "fertilizer": "Apply nitrogen and potassium. Foliar spray with micronutrients.",
        "prevention": "Use certified seeds. Balanced fertilization. Avoid water stress."
    },
    "Brownspot": {
        "description": "Brownspot on rice is caused by Cochliobolus miyabeanus fungus, appearing as oval brown spots.",
        "severity": "Medium",
        "treatment": ["Apply edifenphos or carbendazim", "Seed treatment before planting", "Improve plant nutrition"],
        "fertilizer": "Increase potassium and silicon supply. Correct any nutrient deficiencies.",
        "prevention": "Use disease-free seeds. Proper water management. Field sanitation."
    },
    "Common_Rust": {
        "description": "Common Rust on corn shows as powdery reddish-brown pustules on both leaf surfaces.",
        "severity": "Medium",
        "treatment": ["Apply foliar fungicide (azoxystrobin)", "Spray early when pustules first appear", "Repeat 14 days later"],
        "fertilizer": "Balanced NPK. Adequate potassium strengthens against rust.",
        "prevention": "Plant rust-resistant hybrids. Early planting. Scout regularly."
    },
    "Dried_Leaves": {
        "description": "Dried leaves indicate water stress, root damage, or severe disease infection.",
        "severity": "Medium",
        "treatment": ["Check irrigation system", "Improve water management", "Check roots for rot or damage"],
        "fertilizer": "Apply potassium to improve drought tolerance. Humic acid for root development.",
        "prevention": "Regular irrigation monitoring. Mulching to retain soil moisture."
    },
    "Fusarium": {
        "description": "Fusarium wilt/rot is a soilborne fungal disease causing wilting, root rot, and plant death.",
        "severity": "Very High",
        "treatment": ["Apply trichoderma-based biocontrol", "Soil drench with carbendazim", "Remove and destroy infected plants"],
        "fertilizer": "Apply organic matter to improve soil biology. Avoid excess nitrogen.",
        "prevention": "Crop rotation (4-year cycle). Use resistant varieties. Improve drainage."
    },
    "FusariumFootRot": {
        "description": "Fusarium Foot Rot causes basal stem rot and crown rot in cereal crops.",
        "severity": "High",
        "treatment": ["Seed treatment with thiram + carbendazim", "Soil fumigation if severe", "Remove infected plants"],
        "fertilizer": "Balanced nutrition. Add organic matter to suppress soilborne pathogens.",
        "prevention": "Crop rotation. Deep plowing to bury infected residues. Seed treatment."
    },
    "Grassy_shoot": {
        "description": "Grassy Shoot is a phytoplasma disease of sugarcane causing shoots to look like grass (pale, thin).",
        "severity": "Very High",
        "treatment": ["No cure — remove and destroy infected plants immediately", "Control leafhopper vectors with insecticide", "Use heat-treated setts for planting"],
        "fertilizer": "Healthy plants only: Balanced NPK.",
        "prevention": "Use disease-free planting material. Control leafhoppers. Rogue infected plants."
    },
    "Gray_Leaf_Spot": {
        "description": "Gray Leaf Spot on corn causes rectangular gray-brown lesions bounded by leaf veins.",
        "severity": "High",
        "treatment": ["Apply strobilurin or triazole fungicide", "Spray at tasseling stage", "Improve air circulation by reducing plant density"],
        "fertilizer": "Balanced NPK. Adequate nitrogen but avoid excess.",
        "prevention": "Plant resistant hybrids. Tillage to bury crop residues. Crop rotation."
    },
    "Healthy_Corn_or_Maize_Leaves": {
        "description": "Your corn/maize plant looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue regular monitoring", "Maintain current farm practices"],
        "fertilizer": "Continue balanced NPK fertilization. Top-dress nitrogen at knee-high stage.",
        "prevention": "Keep scouting weekly. Maintain field hygiene. Continue crop rotation."
    },
    "Healthy_Rice_Leaves": {
        "description": "Your rice plant looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue regular monitoring", "Maintain optimal water levels"],
        "fertilizer": "Continue scheduled fertilization. Apply nitrogen in split doses.",
        "prevention": "Regular field scouting. Maintain proper water management."
    },
    "Healthy_Sugercane_Leaves": {
        "description": "Your sugarcane plant looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue current management practices", "Monitor for pests and diseases"],
        "fertilizer": "Apply NPK as per growth stage. Trashing (removal of dry leaves) improves growth.",
        "prevention": "Regular ratoon management. Scout for borers and sucking pests."
    },
    "Healthy_Wheat_Leaves": {
        "description": "Your wheat plant looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue regular monitoring", "Maintain irrigation schedule"],
        "fertilizer": "Apply nitrogen at tillering and jointing stages. Ensure adequate phosphorus.",
        "prevention": "Regular scouting especially in spring. Maintain field sanitation."
    },
    "Healthy_cotton_leaf": {
        "description": "Your cotton leaf looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue regular monitoring", "Scout for bollworms and sucking pests"],
        "fertilizer": "Continue balanced NPK. Foliar spray micronutrients at flowering.",
        "prevention": "IPM practices. Regular scouting for pests. Timely irrigation."
    },
    "Healthy_cotton_plant": {
        "description": "Your cotton plant looks healthy! No disease detected.",
        "severity": "None",
        "treatment": ["Continue current management", "Monitor boll development"],
        "fertilizer": "Potassium at boll formation stage. Balanced nutrition throughout.",
        "prevention": "Regular scouting. Follow IPM guidelines. Timely defoliation."
    },
    "LeafBlight": {
        "description": "Leaf Blight causes large irregular brown lesions that kill leaf tissue rapidly.",
        "severity": "High",
        "treatment": ["Apply propiconazole or mancozeb", "Remove infected leaves", "Reduce humidity by improving spacing"],
        "fertilizer": "Reduce nitrogen. Increase potassium for disease resistance.",
        "prevention": "Crop rotation. Field sanitation. Use resistant varieties."
    },
    "Mealybug": {
        "description": "Mealybugs are white cottony insects that suck plant sap, causing yellowing and stunted growth.",
        "severity": "Medium",
        "treatment": ["Spray imidacloprid or spirotetramat insecticide", "Apply neem oil (5ml/L water)", "Remove heavily infested plant parts"],
        "fertilizer": "Healthy nutrition reduces pest susceptibility. Balanced NPK.",
        "prevention": "Inspect planting material. Avoid ant activity (ants farm mealybugs). Biological control with ladybugs."
    },
    "Pokkah_Boeng": {
        "description": "Pokkah Boeng is a Fusarium disease of sugarcane causing twisted, chlorotic young leaves.",
        "severity": "High",
        "treatment": ["Apply carbendazim or thiophanate-methyl", "Remove infected tops carefully", "Avoid mechanical damage during cultivation"],
        "fertilizer": "Balanced nutrition. Avoid excess nitrogen which increases severity.",
        "prevention": "Use clean planting material. Avoid wounds during intercultural operations."
    },
    "Sett_Rot": {
        "description": "Sett Rot destroys sugarcane seed pieces (setts) before germination, causing poor crop establishment.",
        "severity": "High",
        "treatment": ["Treat setts with carbendazim before planting", "Plant in well-drained soil", "Use healthy setts from disease-free fields"],
        "fertilizer": "Apply organic matter to improve soil drainage and beneficial microbes.",
        "prevention": "Sett treatment is mandatory. Use disease-free seed cane. Avoid waterlogged fields."
    },
    "Tungro": {
        "description": "Tungro is a viral disease of rice transmitted by green leafhoppers, causing yellow-orange discoloration.",
        "severity": "Very High",
        "treatment": ["No direct cure — control leafhopper vectors immediately", "Apply imidacloprid or buprofezin for leafhopper control", "Remove and destroy infected plants"],
        "fertilizer": "Avoid excess nitrogen which attracts leafhoppers. Balanced potassium.",
        "prevention": "Plant resistant varieties (IR64, IR72). Synchronous planting. Control leafhoppers."
    },
    "Viral_Disease": {
        "description": "Viral diseases in crops cause mosaic patterns, stunting, and yield loss. Spread by insect vectors.",
        "severity": "High",
        "treatment": ["Control insect vectors (aphids, whiteflies, leafhoppers)", "Remove and destroy infected plants", "Apply systemic insecticide"],
        "fertilizer": "Maintain plant health with balanced nutrition. Avoid stress conditions.",
        "prevention": "Use virus-free planting material. Control vectors early. Resistant varieties."
    },
    "WheatBlast": {
        "description": "Wheat Blast is an aggressive fungal disease causing bleached heads and severe yield loss.",
        "severity": "Very High",
        "treatment": ["Apply triazole fungicides immediately (tebuconazole, propiconazole)", "Emergency response required", "Report to agricultural authorities"],
        "fertilizer": "Balanced nutrition. Do not apply excess nitrogen.",
        "prevention": "Use resistant varieties. Early planting. Avoid rainy weather at flowering."
    },
    "Yellow_Leaf": {
        "description": "Yellow Leaf in sugarcane is caused by a luteovirus transmitted by aphids, causing leaf yellowing.",
        "severity": "High",
        "treatment": ["Control aphid vectors with imidacloprid", "Remove and destroy infected plants", "Use heat-treated planting material"],
        "fertilizer": "Healthy balanced nutrition. Adequate nitrogen and potassium.",
        "prevention": "Use disease-free setts. Control aphids. Rogue infected plants early."
    },
    "american_bollworm": {
        "description": "American Bollworm (Helicoverpa armigera) is a major pest of cotton, corn, and other crops.",
        "severity": "High",
        "treatment": ["Apply chlorpyrifos or spinosad insecticide", "Use pheromone traps to monitor population", "Apply Bt (Bacillus thuringiensis) as biocontrol"],
        "fertilizer": "Healthy plants tolerate pest pressure better. Balanced NPK.",
        "prevention": "IPM approach. Pheromone traps. Natural enemies conservation. Early planting."
    },
    "diseased_cotton_leaf": {
        "description": "Cotton leaf shows signs of disease — could be bacterial blight, leaf curl virus, or fungal infection.",
        "severity": "High",
        "treatment": ["Identify specific disease for targeted treatment", "Apply broad-spectrum fungicide/bactericide", "Control whitefly and aphid vectors"],
        "fertilizer": "Reduce nitrogen. Apply potassium to boost immunity.",
        "prevention": "Use certified disease-free seeds. Control vectors. Regular scouting."
    },
    "diseased_cotton_plant": {
        "description": "Cotton plant shows systemic disease symptoms affecting overall plant health.",
        "severity": "High",
        "treatment": ["Apply systemic fungicide/bactericide as appropriate", "Improve drainage", "Remove severely infected plants"],
        "fertilizer": "Balanced NPK. Micronutrients especially zinc and boron.",
        "prevention": "Crop rotation. Field sanitation. Resistant varieties. Seed treatment."
    },
    "smut": {
        "description": "Smut is a fungal disease affecting cereal grains, replacing grain contents with black spore masses.",
        "severity": "High",
        "treatment": ["Seed treatment with carboxin + thiram", "Remove smutted heads before spores release", "Burn infected material"],
        "fertilizer": "Healthy balanced nutrition. Seed treatment is most important intervention.",
        "prevention": "Always treat seeds before planting. Use certified clean seed. Crop rotation."
    },
    "thrips": {
        "description": "Thrips are tiny insects that cause silvery streaks and distorted leaves by rasping leaf tissue.",
        "severity": "Medium",
        "treatment": ["Apply spinosad, imidacloprid, or abamectin", "Spray in evening when thrips are most active", "Use blue sticky traps to monitor"],
        "fertilizer": "Healthy balanced nutrition reduces thrips susceptibility.",
        "prevention": "Monitor with sticky traps. Control weeds. Conserve natural predators."
    }
}

# ─── Load Model with Compatibility ───────────────────────────────────────────
print("Loading CropCare AI model...")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

model = None
label_encoder = None
model_config = None

def load_model_safe():
    """Load model with multiple fallback strategies"""
    global model, label_encoder, model_config
    
    # Strategy 1: Direct load
    try:
        print("Attempting to load model...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("✓ Model loaded successfully (without compilation)")
        
        # Try to compile
        try:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Direct load failed: {e}")
        return False

def load_encoder_safe():
    """Load label encoder safely"""
    global label_encoder
    try:
        label_encoder = joblib.load(ENCODER_PATH)
        print(f"✓ Label encoder loaded: {len(label_encoder.classes_)} classes")
        return True
    except Exception as e:
        print(f"✗ Label encoder load failed: {e}")
        return False

def load_config_safe():
    """Load model config safely"""
    global model_config
    try:
        with open(CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
        print(f"✓ Config loaded: {model_config.get('num_classes', 'N/A')} classes")
        if 'test_accuracy' in model_config:
            print(f"  Test accuracy: {model_config['test_accuracy']*100:.2f}%")
        return True
    except Exception as e:
        print(f"✗ Config load failed: {e}")
        return False

# Try to load all components
model_loaded = load_model_safe()
encoder_loaded = load_encoder_safe()
config_loaded = load_config_safe()

if model_loaded and encoder_loaded and config_loaded:
    print("\n✓ All components loaded successfully!")
else:
    print("\n⚠ Some components failed to load. The API will still work,")
    print("  but predictions may not be available.")
    print("\n  Troubleshooting steps:")
    print("  1. Run: pip install --upgrade numpy")
    print("  2. Run: pip install --upgrade tensorflow")
    print("  3. Check if model files exist in:", MODEL_DIR)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    """Preprocess image bytes for model prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_treatment(class_name):
    """Get treatment info for a disease class."""
    # Try exact match first
    if class_name in TREATMENTS:
        return TREATMENTS[class_name]
    # Try partial match
    for key in TREATMENTS:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return TREATMENTS[key]
    # Default treatment
    return {
        "description": f"Disease detected: {class_name.replace('_', ' ')}",
        "severity": "Unknown",
        "treatment": ["Consult a local agricultural expert for specific treatment advice"],
        "fertilizer": "Maintain balanced nutrition. Consult soil test results.",
        "prevention": "Regular monitoring and field scouting recommended."
    }

def get_severity_color(severity):
    """Return color code for severity level."""
    colors = {"None": "#4CAF50", "Low": "#8BC34A", "Medium": "#FF9800",
              "High": "#F44336", "Very High": "#B71C1C", "Unknown": "#9E9E9E"}
    return colors.get(severity, "#9E9E9E")


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "classes": len(label_encoder.classes_) if label_encoder else 0,
        "accuracy": f"{model_config['test_accuracy']*100:.2f}%" if model_config and 'test_accuracy' in model_config else "N/A",
        "version": "1.0.0",
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts: multipart/form-data with 'image' file
          OR JSON with 'image_base64' field
    Returns: JSON with predictions, disease info, and treatments
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs.", "demo_mode": True}), 503

    if label_encoder is None:
        return jsonify({"error": "Label encoder not loaded. Please check server logs."}), 503

    try:
        # ── Get image from request ────────────────────────────────────────────
        image_bytes = None

        if 'image' in request.files:
            # Multipart form upload
            file = request.files['image']
            image_bytes = file.read()

        elif request.is_json and 'image_base64' in request.json:
            # Base64 encoded image
            b64_data = request.json['image_base64']
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            image_bytes = base64.b64decode(b64_data)

        else:
            return jsonify({"error": "No image provided. Send 'image' file or 'image_base64' field."}), 400

        # ── Preprocess & Predict ──────────────────────────────────────────────
        img_array = preprocess_image(image_bytes)
        predictions = model.predict(img_array, verbose=0)[0]

        # Top-5 predictions
        top5_idx    = np.argsort(predictions)[::-1][:5]
        top5_probs  = predictions[top5_idx]
        top5_labels = label_encoder.inverse_transform(top5_idx)

        top_disease    = top5_labels[0]
        top_confidence = float(top5_probs[0])
        is_healthy     = 'healthy' in top_disease.lower()

        # ── Build response ────────────────────────────────────────────────────
        treatment_info = get_treatment(top_disease)

        response = {
            "success": True,
            "top_disease": top_disease,
            "top_disease_display": top_disease.replace('_', ' '),
            "confidence": round(top_confidence * 100, 2),
            "is_healthy": is_healthy,
            "severity": treatment_info["severity"],
            "severity_color": get_severity_color(treatment_info["severity"]),
            "description": treatment_info["description"],
            "treatment": treatment_info["treatment"],
            "fertilizer": treatment_info["fertilizer"],
            "prevention": treatment_info["prevention"],
            "top5_predictions": [
                {
                    "disease": str(top5_labels[i]),
                    "disease_display": str(top5_labels[i]).replace('_', ' '),
                    "confidence": round(float(top5_probs[i]) * 100, 2)
                }
                for i in range(len(top5_labels))
            ]
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return list of all detectable diseases."""
    if label_encoder is None:
        return jsonify({"error": "Label encoder not loaded"}), 503

    classes = []
    for cls in label_encoder.classes_:
        treatment = get_treatment(cls)
        classes.append({
            "name": cls,
            "display": cls.replace('_', ' '),
            "severity": treatment["severity"],
            "is_healthy": 'healthy' in cls.lower()
        })

    return jsonify({"classes": classes, "total": len(classes)})


@app.route('/predict_demo', methods=['POST'])
def predict_demo():
    """Demo prediction endpoint for testing without model"""
    return jsonify({
        "success": True,
        "demo_mode": True,
        "top_disease": "Healthy_Wheat_Leaves",
        "top_disease_display": "Healthy Wheat Leaves",
        "confidence": 95.5,
        "is_healthy": True,
        "severity": "None",
        "severity_color": "#4CAF50",
        "description": "Your wheat plant looks healthy! No disease detected.",
        "treatment": ["Continue regular monitoring", "Maintain irrigation schedule"],
        "fertilizer": "Apply nitrogen at tillering and jointing stages. Ensure adequate phosphorus.",
        "prevention": "Regular scouting especially in spring. Maintain field sanitation.",
        "top5_predictions": [
            {"disease": "Healthy_Wheat_Leaves", "disease_display": "Healthy Wheat Leaves", "confidence": 95.5},
            {"disease": "BrownRust", "disease_display": "Brown Rust", "confidence": 2.3},
            {"disease": "Blast", "disease_display": "Blast", "confidence": 1.2},
            {"disease": "Blight", "disease_display": "Blight", "confidence": 0.8},
            {"disease": "Smut", "disease_display": "Smut", "confidence": 0.2}
        ]
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CropCare AI Backend Server")
    print("="*55)
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  NumPy version      : {np.__version__}")
    print(f"  Model directory    : {MODEL_DIR}")
    print(f"  Model loaded       : {model is not None}")
    print(f"  Running on         : http://0.0.0.0:5000")
    print(f"  For Android        : http://<your-PC-IP>:5000")
    print("  Endpoints:")
    print("    GET  /health        - Server status")
    print("    POST /predict       - Disease detection")
    print("    POST /predict_demo  - Demo detection (no model required)")
    print("    GET  /classes       - All disease classes")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=os.environ.get('FLASK_DEBUG') == '1')