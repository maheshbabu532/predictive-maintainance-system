from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import crud
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base
from schemas import UserCreate, UserLogin, UserResponse, Token
from crud import create_user, get_user_by_username, verify_password, get_user_by_email

# Initialize database
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for integration with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # Adjust this for production to specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/check-availability")
def check_availability(username: str = None, email: str = None, db: Session = Depends(get_db)):
    """
    Check if a username or email already exists in the database.
    """
    if username:
        existing_user = get_user_by_username(db, username)
        if existing_user:
            return {"exists": True, "field": "username"}

    if email:
        existing_email = get_user_by_email(db, email)
        if existing_email:
            return {"exists": True, "field": "email"}

    return {"exists": False}

@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username already exists
    if get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")

    # Check if email already exists
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already exists")

    # Create the user if both username and email are unique
    new_user = create_user(db, user.username, user.email, user.password)
    return {"message": "User registered successfully", "user_id": new_user.id}

@app.post("/token", response_model=Token)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, user.username)
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    access_token = crud.create_access_token(data={"sub": db_user.username}, expires_delta=timedelta(minutes=crud.ACCESS_TOKEN_EXPIRE_MINUTES))
    
    return {"access_token": access_token, "token_type": "bearer"}

     # Create JWT token
    access_token = crud.create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected route (requires authentication)
@app.get("/protected")
def protected_route(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = crud.verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expiredtoken")

    username = payload.get("sub")
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {"message": f"Hello {user.username}, you accessed a protected route!"}

# Ensure necessary directories exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Request model for prediction input
class PredictRequest(BaseModel):
    machine: str
    parameters: dict

@app.get("/")
async def root():
    return {"message": "Welcome to the Predictive Maintenance Backend API!"}

# Route to upload dataset
@app.post("/upload-dataset/")
async def upload_dataset(machine: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    dataset_path = f"/Users/mahes/OneDrive/Desktop/predictive/PdM_project/datasets/{machine.replace(' ', '_')}.csv"
    with open(dataset_path, "wb") as f:
        f.write(file.file.read())
    return {"message": f"Dataset uploaded for {machine}", "path": dataset_path}

# Route to train model
@app.post("/train-model/")
async def train_model(machine: str = Form(...)):
    dataset_path = f"/Users/mahes/OneDrive/Desktop/predictive/PdM_project/datasets/{machine.replace(' ', '_')}.csv"

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"No dataset found for {machine}")

    # Load dataset
    df = pd.read_csv(dataset_path)

    if "Target" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must contain 'Target' column")

    # Separate features and target
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save model
    model_path = f"/Users/mahes/OneDrive/Desktop/predictive/PdM_project/models/{machine.replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)

    return {"message": f"Model trained for {machine}", "accuracy": accuracy, "model_path": model_path}

# Route to predict failure probability
@app.post("/predict/")
async def predict(request: PredictRequest):
    machine = request.machine
    parameters = request.parameters

    # Load the trained model
    model_path = f"/Users/mahes/OneDrive/Desktop/predictive/PdM_project/models/{machine.replace(' ', '_')}_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model not trained yet")

    model = joblib.load(model_path)

     # Define parameter ranges for dynamic alert generation
    parameter_details = {
        "Lathe Machine": {
            "Spindle Speed": {
                "range": (1200, 2000),
                "alert": "Spindle speed is high. Please conduct maintenance surveys to check for overheating or misalignment."
            },
            "Tool Pressure": {
                "range": (5.5, 6.0),
                "alert": "Tool pressure is outside the range. Inspect for tool wear or improper settings."
            },
            "Cutting Temperature": {
                "range": (70, 80),
                "alert": "Cutting temperature is high. Verify coolant levels and cutting tool conditions."
            },
            "Vibration": {
                "range": (0.5, 1.0),
                "alert": "High vibration detected. Check for unbalanced components or loose parts."
            }
        },
        "Milling Machine": {
            "Tool Speed": {
                "range": (1000, 4000),
                "alert": "Tool speed is excessive. Inspect tool sharpness and calibration."
            },
            "Table Feed": {
                "range": (150, 300),
                "alert": "Table feed is outside acceptable range. Evaluate machine settings for accuracy."
            },
            "Coolant Flow": {
                "range": (1, 10),
                "alert": "Coolant flow is insufficient. Check for blockages or coolant supply issues."
            },
            "Cutting Depth": {
                "range": (1.0, 2.9),
                "alert": "Cutting depth exceeds limits. Reduce depth to avoid tool breakage."
            }
        },
        "Drill Machine": {
            "Drill Speed": {
                "range": (300, 1500),
                "alert": "Drill speed is out of range. Verify drill settings and material compatibility."
            },
            "Pressure": {
                "range": (50, 200),
                "alert": "Pressure exceeds recommended levels. Check for system overloading."
            },
            "Material Hardness": {
                "range": (30, 60),
                "alert": "Material hardness is not suitable for drilling. Reassess material selection."
            },
            "Coolant Level": {
                "range": (0.5, 5),
                "alert": "Coolant level is insufficient. Refill or inspect coolant system."
            }
        },
        "Grinder": {
            "Grinding Wheel Speed": {
                "range": (1000, 3000),
                "alert": "Grinding wheel speed is excessive. Inspect wheel for wear or imbalance."
            },
            "Wheel Pressure": {
                "range": (150, 400),
                "alert": "Wheel pressure is out of range. Ensure proper alignment and force control."
            },
            "Coolant Flow": {
                "range": (2, 8),
                "alert": "Coolant flow is inadequate. Verify flow rate and check for blockages."
            },
            "Temperature": {
                "range": (110, 150),
                "alert": "Temperature is too high. Reduce load or inspect cooling system."
            }
        },
        "Injection Molding Machine": {
            "Injection Pressure": {
                "range": (500, 2000),
                "alert": "Injection pressure is too high. Inspect mold and nozzle for clogs."
            },
            "Mold Temperature": {
                "range": (30, 120),
                "alert": "Mold temperature is out of range. Adjust heating or cooling settings."
            },
            "Cycle Time": {
                "range": (10, 30),
                "alert": "Cycle time exceeds expected range. Check for system inefficiencies."
            },
            "Clamping Force": {
                "range": (50, 300),
                "alert": "Clamping force is inadequate. Reassess clamp settings and mold alignment."
            }
        },
        "CNC Router": {
            "Spindle Speed": {
                "range": (8000, 24000),
                "alert": "Spindle speed is too high. Inspect tool balance and settings."
            },
            "Feed Rate": {
                "range": (100, 500),
                "alert": "Feed rate is outside safe limits. Adjust to prevent tool wear."
            },
            "Tool Diameter": {
                "range": (3, 12),
                "alert": "Tool diameter is unsuitable for the operation. Reevaluate tool selection."
            },
            "Bed Temperature": {
                "range": (60, 110),
                "alert": "Bed temperature is out of range. Inspect heating system for issues."
            }
        },
        "Press Machine": {
            "Press Force": {
                "range": (100, 1000),
                "alert": "Press force exceeds safety limits. Check hydraulic system pressure."
            },
            "Stroke Length": {
                "range": (50, 200),
                "alert": "Stroke length is excessive. Inspect mechanical stops and settings."
            },
            "Cycle Time": {
                "range": (5, 30),
                "alert": "Cycle time is too long. Investigate delays in press operation."
            },
            "Die Temperature": {
                "range": (100, 500),
                "alert": "Die temperature is out of acceptable range. Inspect heating system."
            }
        },
        "3D Printer": {
            "Nozzle Temperature": {
                "range": (190, 240),
                "alert": "Nozzle temperature is out of range. Ensure consistent filament melting."
            },
            "Bed Temperature": {
                "range": (60, 100),
                "alert": "Bed temperature is incorrect. Adjust to improve print adhesion."
            },
            "Layer Thickness": {
                "range": (0.1, 0.3),
                "alert": "Layer thickness is outside recommended limits. Adjust slicing settings."
            },
            "Print Speed": {
                "range": (40, 150),
                "alert": "Print speed is too high. Reduce speed to improve print quality."
            }
        },
        "Compressor": {
            "Air Pressure": {
                "range": (6, 12),
                "alert": "Air pressure is out of range. Inspect compressor settings."
            },
            "Motor Speed": {
                "range": (800, 3000),
                "alert": "Motor speed is too high. Check motor for overheating or faults."
            },
            "Oil Temperature": {
                "range": (30, 90),
                "alert": "Oil temperature is outside safe limits. Verify lubrication system."
            },
            "Flow Rate": {
                "range": (50, 200),
                "alert": "Flow rate is inadequate. Inspect for obstructions in the air system."
            }
        },
        "Pump": {
            "Flow Rate": {
                "range": (10, 200),
                "alert": "Flow rate is outside acceptable range. Check pump and piping system."
            },
            "Pump Speed": {
                "range": (500, 3000),
                "alert": "Pump speed is too high. Inspect for overloading or wear."
            },
            "Discharge Pressure": {
                "range": (5, 50),
                "alert": "Discharge pressure is excessive. Verify pump and valve settings."
            },
            "Fluid Temperature": {
                "range": (20, 80),
                "alert": "Fluid temperature is too high. Check for cooling system faults."
            }
        },
        "HVAC System": {
            "Air Flow Rate": {
                "range": (200, 5000),
                "alert": "Air flow rate is too low. Inspect for blockages or fan issues."
            },
            "Cooling Efficiency": {
                "range": (2.5, 6),
                "alert": "Cooling efficiency is suboptimal. Check refrigerant levels."
            },
            "Compressor Pressure": {
                "range": (5, 15),
                "alert": "Compressor pressure is out of range. Inspect for leaks."
            },
            "Refrigerant Temperature": {
                "range": (0, 50),
                "alert": "Refrigerant temperature is too high. Verify cooling circuit."
            }
        },
        "Boiler": {
            "Steam Pressure": {
                "range": (1, 20),
                "alert": "Steam pressure is outside safe limits. Inspect safety valves."
            },
            "Water Temperature": {
                "range": (60, 120),
                "alert": "Water temperature is too high. Check heating elements."
            },
            "Fuel Flow Rate": {
                "range": (1, 10),
                "alert": "Fuel flow rate is insufficient. Verify fuel supply system."
            },
            "Exhaust Temperature": {
                "range": (100, 300),
                "alert": "Exhaust temperature is excessive. Inspect for combustion inefficiency."
            }
        },
        "Water Jet Cutter": {
            "Water Pressure": {
                "range": (3000, 6000),
                "alert": "Water pressure is outside the optimal range. Inspect the pump and connections."
            },
            "Nozzle Diameter": {
                "range": (0.1, 0.5),
                "alert": "Nozzle diameter is incorrect. Verify nozzle selection for desired precision."
            },
            "Abrasive Flow Rate": {
                "range": (0.1, 1.0),
                "alert": "Abrasive flow rate is insufficient. Check hopper or abrasive supply system."
            },
            "Cutting Speed": {
                "range": (50, 500),
                "alert": "Cutting speed is out of range. Adjust feed rate for optimal cutting quality."
            }
        },
        "Laser Cutting Machine": {
            "Laser Power": {
                "range": (100, 3000),
                "alert": "Laser power is outside acceptable levels. Check power supply and laser alignment."
            },
            "Cutting Speed": {
                "range": (10, 1000),
                "alert": "Cutting speed exceeds safety limits. Adjust for material compatibility."
            },
            "Focus Position": {
                "range": (0, 2),
                "alert": "Focus position is misaligned. Recalibrate for precision cutting."
            },
            "Assist Gas Pressure": {
                "range": (0.5, 15),
                "alert": "Assist gas pressure is inadequate. Check gas supply and regulator settings."
            }
        },
        "CNC Milling Machine": {
            "Spindle Speed": {
                "range": (8000, 24000),
                "alert": "Spindle speed is too high. Inspect for potential tool wear or breakage."
            },
            "Feed Rate": {
                "range": (100, 600),
                "alert": "Feed rate is outside the acceptable range. Reassess settings for material type."
            },
            "Tool Diameter": {
                "range": (2, 16),
                "alert": "Tool diameter is incompatible. Ensure correct tool selection for operation."
            },
            "Coolant Temperature": {
                "range": (20, 60),
                "alert": "Coolant temperature is outside optimal range. Inspect cooling system."
            }
        },
        "Extrusion Machine": {
            "Barrel Temperature": {
                "range": (150, 300),
                "alert": "Barrel temperature is excessive. Check heating elements and control systems."
            },
            "Screw Speed": {
                "range": (20, 150),
                "alert": "Screw speed is outside acceptable levels. Adjust for optimal material flow."
            },
            "Melt Pressure": {
                "range": (100, 3000),
                "alert": "Melt pressure is too high. Inspect material flow and die alignment."
            },
            "Die Temperature": {
                "range": (180, 250),
                "alert": "Die temperature is out of range. Adjust heating system settings."
            }
        },
        "Wire EDM Machine": {
            "Wire Tension": {
                "range": (5, 20),
                "alert": "Wire tension is insufficient. Adjust for proper cutting accuracy."
            },
            "Cutting Speed": {
                "range": (10, 100),
                "alert": "Cutting speed is too high. Reduce speed to avoid wire breakage."
            },
            "Spark Voltage": {
                "range": (50, 100),
                "alert": "Spark voltage is outside the recommended range. Check power supply."
            },
            "Dielectric Flow": {
                "range": (5, 20),
                "alert": "Dielectric flow rate is insufficient. Verify pump and fluid levels."
            }
        },
        "Crane": {
            "Lifting Capacity": {
                "range": (1, 50),
                "alert": "Lifting capacity exceeds limits. Reduce load to prevent mechanical failure."
            },
            "Boom Length": {
                "range": (5, 50),
                "alert": "Boom length is out of range. Inspect boom settings and stability."
            },
            "Rotation Speed": {
                "range": (0, 3),
                "alert": "Rotation speed is too high. Reduce to avoid tipping risks."
            },
            "Hoist Speed": {
                "range": (1, 20),
                "alert": "Hoist speed is excessive. Slow down to ensure safety and precision."
            }
        },
        "Forklift": {
            "Lifting Capacity": {
                "range": (1, 20),
                "alert": "Lifting capacity is too high. Check load and balance."
            },
            "Mast Height": {
                "range": (3, 12),
                "alert": "Mast height is beyond the safe range. Ensure stable operation."
            },
            "Engine Temperature": {
                "range": (70, 110),
                "alert": "Engine temperature is too high. Check cooling system and oil levels."
            },
            "Fuel Level": {
                "range": (10, 100),
                "alert": "Fuel level is too low. Refuel to maintain operation."
            }
        },
        "Heat Exchanger": {
            "Inlet Temperature": {
                "range": (50, 150),
                "alert": "Inlet temperature is out of range. Adjust heat exchanger settings."
            },
            "Outlet Temperature": {
                "range": (30, 120),
                "alert": "Outlet temperature is too low. Ensure heat exchange process is efficient."
            },
            "Flow Rate": {
                "range": (10, 500),
                "alert": "Flow rate is below optimal. Inspect pumps and flow control valves."
            },
            "Heat Transfer Efficiency": {
                "range": (60, 95),
                "alert": "Heat transfer efficiency is insufficient. Check for fouling or system leaks."
            }
        },
        "Conveyor System": {
            "Belt Speed": {
                "range": (1, 10),
                "alert": "Belt speed is too high. Adjust to prevent product misalignment."
            },
            "Load Weight": {
                "range": (10, 1000),
                "alert": "Load weight exceeds maximum capacity. Adjust for safer operation."
            },
            "Motor Temperature": {
                "range": (30, 80),
                "alert": "Motor temperature is too high. Inspect for overheating and motor load."
            },
            "Power Consumption": {
                "range": (1, 50),
                "alert": "Power consumption is too high. Check system efficiency and energy usage."
            }
        },
        "Packaging Machine": {
            "Sealing Temperature": {
                "range": (100, 250),
                "alert": "Sealing temperature is incorrect. Adjust to ensure proper sealing."
            },
            "Cycle Time": {
                "range": (2, 20),
                "alert": "Cycle time is too short. Ensure adequate cooling and sealing."
            },
            "Film Thickness": {
                "range": (0.5, 3),
                "alert": "Film thickness is incorrect. Adjust for packaging material specifications."
            },
            "Material Feed Rate": {
                "range": (10, 500),
                "alert": "Material feed rate is too high. Adjust to avoid overloading the machine."
            }
        },
        "Robotic Arm": {
            "Axis Speed": {
                "range": (30, 180),
                "alert": "Axis speed is too high. Reduce for smoother operation and precision."
            },
            "Payload Capacity": {
                "range": (1, 500),
                "alert": "Payload capacity exceeds limits. Reduce load to prevent damage."
            },
            "Joint Angle": {
                "range": (0, 360),
                "alert": "Joint angle is misaligned. Adjust for optimal movement."
            },
            "Torque": {
                "range": (10, 200),
                "alert": "Torque exceeds the limit. Check for proper motor calibration."
            }
        },
        "Vacuum Forming Machine": {
            "Vacuum Pressure": {
                "range": (0.1, 1.0),
                "alert": "Vacuum pressure is inadequate. Inspect pump and seals."
            },
            "Heating Temperature": {
                "range": (100, 250),
                "alert": "Heating temperature is too high. Reduce to avoid material degradation."
            },
            "Forming Time": {
                "range": (10, 60),
                "alert": "Forming time is too long. Adjust for optimal molding quality."
            },
            "Sheet Thickness": {
                "range": (0.5, 5),
                "alert": "Sheet thickness is outside acceptable limits. Verify material type."
            }
        },
        "Hydraulic Press": {
            "Hydraulic Pressure": {
                "range": (50, 300),
                "alert": "Hydraulic pressure exceeds safe limits. Check for overloading or malfunction."
            },
            "Ram Speed": {
                "range": (10, 150),
                "alert": "Ram speed is too high. Slow down to avoid damage to workpiece."
            },
            "Cycle Time": {
                "range": (5, 60),
                "alert": "Cycle time is too short. Ensure complete pressing before release."
            },
            "Oil Temperature": {
                "range": (30, 90),
                "alert": "Oil temperature is outside the optimal range. Inspect cooling system."
            }
        },
        "Industrial Oven": {
            "Oven Temperature": {
                "range": (50, 300),
                "alert": "Oven temperature is too high or low. Adjust to optimal levels for proper material curing."
            },
            "Air Circulation Speed": {
                "range": (1, 10),
                "alert": "Air circulation speed is too low or high. Ensure uniform heating by adjusting fan speed."
            },
            "Humidity Level": {
                "range": (10, 90),
                "alert": "Humidity level is out of range. Adjust to maintain desired environmental conditions."
            },
            "Energy Consumption": {
                "range": (1, 50),
                "alert": "Energy consumption is too high. Inspect heating elements and insulation."
            }
        },
        "Food Processing Machine": {
            "Motor Speed": {
                "range": (500, 3000),
                "alert": "Motor speed is out of range. Adjust to avoid damage to components or incorrect processing."
            },
            "Processing Temperature": {
                "range": (20, 100),
                "alert": "Processing temperature is too high or low. Ensure consistent cooking or freezing."
            },
            "Feed Rate": {
                "range": (50, 500),
                "alert": "Feed rate exceeds or is below the optimal range. Adjust for proper material processing."
            },
            "Pressure": {
                "range": (5, 50),
                "alert": "Pressure is too high or low. Inspect valve and pressure control system."
            }
        },
        "Metal Shearing Machine": {
            "Shear Force": {
                "range": (10, 200),
                "alert": "Shear force is too high or low. Ensure optimal settings for clean cuts."
            },
            "Blade Gap": {
                "range": (0.1, 5),
                "alert": "Blade gap is too wide or narrow. Adjust for proper shearing precision."
            },
            "Cutting Speed": {
                "range": (5, 100),
                "alert": "Cutting speed is too fast or slow. Adjust for desired material shearing."
            },
            "Sheet Thickness": {
                "range": (1, 10),
                "alert": "Sheet thickness exceeds the cutting capacity. Ensure material is within the machine’s capacity."
            }
        },
        "Injection Stretch Blow Molding Machine": {
            "Injection Pressure": {
                "range": (100, 2000),
                "alert": "Injection pressure is too high or low. Adjust to ensure accurate molding."
            },
            "Mold Temperature": {
                "range": (30, 120),
                "alert": "Mold temperature is out of range. Ensure proper cooling or heating of mold."
            },
            "Stretch Speed": {
                "range": (1, 10),
                "alert": "Stretch speed is too fast or slow. Adjust to ensure uniform molding."
            },
            "Cycle Time": {
                "range": (10, 30),
                "alert": "Cycle time is too short or long. Optimize for maximum production efficiency."
            }
        },
        "Vacuum Pump": {
            "Vacuum Level": {
                "range": (0.1, 1.0),
                "alert": "Vacuum level is too low. Inspect pump for leaks or blockage."
            },
            "Motor Speed": {
                "range": (500, 3000),
                "alert": "Motor speed is too high or low. Adjust for efficient pump operation."
            },
            "Oil Temperature": {
                "range": (30, 80),
                "alert": "Oil temperature is excessive. Check lubrication system for faults."
            },
            "Flow Rate": {
                "range": (10, 200),
                "alert": "Flow rate is out of range. Inspect vacuum pump components for performance."
            }
        },
        "Centrifugal Fan": {
            "Fan Speed": {
                "range": (500, 3000),
                "alert": "Fan speed is too high or low. Adjust for optimal air flow."
            },
            "Air Flow Rate": {
                "range": (1000, 50000),
                "alert": "Air flow rate is insufficient. Inspect ducts and filters for obstruction."
            },
            "Static Pressure": {
                "range": (5, 200),
                "alert": "Static pressure is out of range. Check fan motor and ducting for restrictions."
            },
            "Vibration": {
                "range": (0.5, 2.0),
                "alert": "Vibration levels are too high. Check for imbalance in the fan blades or motor."
            }
        },
        "Cooling Tower": {
            "Water Flow Rate": {
                "range": (500, 5000),
                "alert": "Water flow rate is too high or low. Inspect water pumps for performance."
            },
            "Air Flow Rate": {
                "range": (1000, 30000),
                "alert": "Air flow rate is not optimal. Adjust fan speed or check air intake."
            },
            "Inlet Water Temperature": {
                "range": (35, 50),
                "alert": "Inlet water temperature exceeds optimal levels. Adjust cooling system."
            },
            "Outlet Water Temperature": {
                "range": (25, 40),
                "alert": "Outlet water temperature is incorrect. Inspect heat exchange efficiency."
            }
        },
        "Diesel Generator": {
            "Fuel Consumption": {
                "range": (10, 50),
                "alert": "Fuel consumption is too high. Check fuel system for leaks or inefficiencies."
            },
            "Engine Speed": {
                "range": (1500, 3000),
                "alert": "Engine speed is abnormal. Inspect governor and fuel system."
            },
            "Oil Temperature": {
                "range": (70, 110),
                "alert": "Oil temperature is out of range. Check for potential engine overheating."
            },
            "Power Output": {
                "range": (10, 500),
                "alert": "Power output is insufficient. Check generator capacity and fuel levels."
            }
        },
        "Electrical Transformer": {
            "Load Current": {
                "range": (100, 2000),
                "alert": "Load current exceeds safe operating levels. Reduce load or check for faults."
            },
            "Oil Temperature": {
                "range": (40, 100),
                "alert": "Oil temperature is too high. Check for inadequate cooling or oil circulation."
            },
            "Voltage Regulation": {
                "range": (0.5, 5),
                "alert": "Voltage regulation is out of range. Check transformer settings and load conditions."
            },
            "Core Loss": {
                "range": (1, 10),
                "alert": "Core loss exceeds limits. Check transformer efficiency and load conditions."
            }
        },
        "Automatic Welding Machine": {
            "Welding Current": {
                "range": (50, 500),
                "alert": "Welding current is too high or low. Adjust for consistent welds."
            },
            "Welding Voltage": {
                "range": (10, 50),
                "alert": "Welding voltage is outside the recommended range. Inspect machine settings."
            },
            "Wire Feed Rate": {
                "range": (1, 10),
                "alert": "Wire feed rate is too fast or slow. Adjust to match welding speed and material."
            },
            "Gas Flow Rate": {
                "range": (5, 25),
                "alert": "Gas flow rate is insufficient. Check gas supply and regulator settings."
            }
        },
        "Shaper Machine": {
            "Cutting Speed": {
                "range": (10, 100),
                "alert": "Cutting speed is too high or low. Adjust for optimal cutting precision."
            },
            "Stroke Length": {
                "range": (50, 700),
                "alert": "Stroke length is out of range. Adjust for desired cut depth and workpiece size."
            },
            "Feed Rate": {
                "range": (0.2, 2),
                "alert": "Feed rate is too high or low. Adjust for smooth and precise operation."
            },
            "Tool Pressure": {
                "range": (5, 20),
                "alert": "Tool pressure is abnormal. Check tool wear or adjustment."
            }
        },
        "Planer Machine": {
            "Cutting Speed": {
                "range": (10, 150),
                "alert": "Cutting speed is too high or low. Adjust for accurate material removal."
            },
            "Table Size": {
                "range": (1, 5),
                "alert": "Table size is too small or large for the material being processed."
            },
            "Feed Rate": {
                "range": (0.2, 2),
                "alert": "Feed rate is incorrect. Adjust to prevent material damage or poor surface finish."
            },
            "Vibration Level": {
                "range": (0.1, 1.0),
                "alert": "Excessive vibration detected. Check machine foundation or alignment."
            }
        },
        "Blow Molding Machine": {
            "Blow Pressure": {
                "range": (2, 15),
                "alert": "Blow pressure is too high or low. Adjust for uniform mold formation."
            },
            "Mold Temperature": {
                "range": (30, 150),
                "alert": "Mold temperature is incorrect. Ensure optimal heating for quality molding."
            },
            "Cycle Time": {
                "range": (10, 60),
                "alert": "Cycle time is too long or short. Adjust for optimal production efficiency."
            },
            "Air Flow Rate": {
                "range": (100, 1000),
                "alert": "Air flow rate is insufficient. Ensure proper cooling and molding pressure."
            }
        },
        "Roller Press": {
            "Roller Pressure": {
                "range": (500, 3000),
                "alert": "Roller pressure is too high or low. Adjust to maintain material flow."
            },
            "Roller Speed": {
                "range": (50, 300),
                "alert": "Roller speed is too fast or slow. Ensure uniform pressing for material consistency."
            },
            "Feed Rate": {
                "range": (1, 20),
                "alert": "Feed rate is too fast or slow. Adjust for efficient processing."
            },
            "Temperature": {
                "range": (30, 150),
                "alert": "Temperature is out of range. Inspect heating system for efficiency."
            }
        },
        "Vibrating Screen": {
            "Vibration Amplitude": {
                "range": (2, 10),
                "alert": "Vibration amplitude is excessive. Adjust to prevent damage to material or screen."
            },
            "Screening Speed": {
                "range": (200, 1000),
                "alert": "Screening speed is too high or low. Adjust for desired material separation."
            },
            "Material Feed Rate": {
                "range": (1, 50),
                "alert": "Material feed rate is too high or low. Adjust to match screen capacity."
            },
            "Motor Power": {
                "range": (1, 10),
                "alert": "Motor power is insufficient or excessive. Check motor load and capacity."
            }
        },
        "Hammer Mill": {
            "Rotor Speed": {
                "range": (1000, 3000),
                "alert": "Rotor speed is too high or low. Adjust for uniform grinding."
            },
            "Feed Rate": {
                "range": (1, 20),
                "alert": "Feed rate is too high or low. Ensure consistent material processing."
            },
            "Hammer Wear": {
                "range": (0.1, 5),
                "alert": "Hammer wear is excessive. Inspect and replace hammers to maintain efficiency."
            },
            "Output Size": {
                "range": (0.1, 10),
                "alert": "Output size is inconsistent. Adjust grinding parameters for desired particle size."
            }
        },
        "Industrial Crusher": {
            "Crusher Speed": {
                "range": (500, 2000),
                "alert": "Crusher speed is too high or low. Ensure optimal material crushing."
            },
            "Feed Rate": {
                "range": (1, 50),
                "alert": "Feed rate is too high or low. Adjust to prevent overloading the crusher."
            },
            "Output Size": {
                "range": (0.5, 50),
                "alert": "Output size is inconsistent. Check crushing mechanism for optimal output."
            },
            "Motor Power": {
                "range": (10, 200),
                "alert": "Motor power is insufficient or excessive. Check motor specifications."
            }
        },
        "Concrete Mixer": {
            "Drum Speed": {
                "range": (10, 25),
                "alert": "Drum speed is too high or low. Adjust for proper mixing of materials."
            },
            "Mixing Time": {
                "range": (30, 300),
                "alert": "Mixing time is too short or long. Ensure uniform consistency of the mix."
            },
            "Batch Volume": {
                "range": (50, 500),
                "alert": "Batch volume is too large or small. Adjust for optimal mix consistency."
            },
            "Motor Power": {
                "range": (1, 10),
                "alert": "Motor power is too low or high. Ensure proper motor size for the mixer's capacity."
            }
        },
        "Road Roller": {
            "Drum Pressure": {
                "range": (20, 200),
                "alert": "Drum pressure is too high or low. Adjust for optimal compaction performance."
            },
            "Speed": {
                "range": (2, 12),
                "alert": "Speed is too high or low. Adjust to prevent uneven compaction or damage."
            },
            "Vibration Frequency": {
                "range": (20, 50),
                "alert": "Vibration frequency is out of range. Ensure efficient soil or asphalt compaction."
            },
            "Fuel Consumption": {
                "range": (5, 20),
                "alert": "Fuel consumption is higher than expected. Check for engine performance issues."
            }
        },
        "Earth Moving Machine": {
            "Bucket Capacity": {
                "range": (0.5, 5),
                "alert": "Bucket capacity is too high or low for the material. Adjust for optimal load management."
            },
            "Engine Power": {
                "range": (50, 500),
                "alert": "Engine power is too low or high for the work requirements."
            },
            "Hydraulic Pressure": {
                "range": (150, 300),
                "alert": "Hydraulic pressure is not within optimal range. Check for leaks or maintenance issues."
            },
            "Cycle Time": {
                "range": (10, 50),
                "alert": "Cycle time is too short or long. Adjust to improve efficiency and reduce wear."
            }
        },
        "CNC Turning Center": {
            "Spindle Speed": {
                "range": (500, 5000),
                "alert": "Spindle speed is too high or low. Adjust for smooth machining."
            },
            "Feed Rate": {
                "range": (0.1, 5),
                "alert": "Feed rate is too high or low. Ensure precise cutting and material handling."
            },
            "Cutting Temperature": {
                "range": (50, 150),
                "alert": "Cutting temperature is too high. Check coolant or cutting speed to avoid damage."
            },
            "Tool Wear": {
                "range": (0.01, 0.5),
                "alert": "Tool wear is excessive. Replace tool for consistent quality."
            }
        },
        "Metal Bending Machine": {
            "Bending Force": {
                "range": (50, 500),
                "alert": "Bending force is too high or low. Adjust for proper metal bending."
            },
            "Bend Angle": {
                "range": (0, 180),
                "alert": "Bend angle is incorrect. Ensure correct angle for accurate metal shaping."
            },
            "Tool Speed": {
                "range": (10, 100),
                "alert": "Tool speed is too high or low. Adjust for consistent bending force."
            },
            "Material Thickness": {
                "range": (0.5, 10),
                "alert": "Material thickness is not within the machine’s capability. Check for material compatibility."
            }
        },
        "Thread Rolling Machine": {
            "Rolling Force": {
                "range": (10, 200),
                "alert": "Rolling force is too high or low. Ensure smooth thread formation."
            },
            "Spindle Speed": {
                "range": (100, 3000),
                "alert": "Spindle speed is incorrect. Adjust for precise thread rolling."
            },
            "Thread Pitch": {
                "range": (0.5, 5),
                "alert": "Thread pitch is out of range. Ensure correct threading specifications."
            },
            "Material Hardness": {
                "range": (30, 60),
                "alert": "Material hardness is not optimal for the rolling process. Check material selection."
            }
        },
        "Electric Arc Furnace": {
            "Arc Voltage": {
                "range": (200, 500),
                "alert": "Arc voltage is out of range. Check the electrode and power supply."
            },
            "Current": {
                "range": (10, 100),
                "alert": "Current is too high or low. Adjust to maintain stable arc and melting."
            },
            "Temperature": {
                "range": (1200, 1600),
                "alert": "Temperature is outside optimal range. Monitor for over- or under-heating of the melt."
            },
            "Energy Consumption": {
                "range": (50, 300),
                "alert": "Energy consumption is unusually high. Check for inefficiencies or process irregularities."
            }
        },
        "Induction Furnace": {
            "Melting Temperature": {
                "range": (1000, 1600),
                "alert": "Melting temperature is too high or low. Ensure proper heat for the material."
            },
            "Power Input": {
                "range": (50, 500),
                "alert": "Power input is inconsistent. Check power supply and coil system."
            },
            "Melting Time": {
                "range": (30, 120),
                "alert": "Melting time is too long or short. Adjust for consistent material properties."
            },
            "Coolant Flow Rate": {
                "range": (10, 50),
                "alert": "Coolant flow rate is too high or low. Ensure cooling efficiency for furnace operation."
            }
        }
    }

    # Convert parameters to DataFrame
    try:
        input_data = pd.DataFrame([parameters])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter structure: {str(e)}")

    # Check feature compatibility
    missing_features = [col for col in model.feature_names_in_ if col not in input_data.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")

    # Predict failure probability
    try:
        failure_probability = model.predict_proba(input_data)[0][1]  # Probability of failure
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Generate dynamic alerts
    alert_messages = []
    if machine in parameter_details:
        for param, value in parameters.items():
            if param in parameter_details[machine]:
                try:
                    value = float(value)
                    min_val, max_val = parameter_details[machine][param]["range"]
                    alert_message = parameter_details[machine][param]["alert"]
                    threshold = max_val - (0.5 * (max_val - min_val))
                    print(f"Checking {param}: Value={value}, Range={min_val}-{max_val}, Threshold={threshold}")
                    if value > threshold:
                        deviation = ((max_val - value) / (max_val - min_val)) * 100
                        alert_messages.append(
                            f"{alert_message} (Value: {value}, Deviation: {deviation:.1f}% above the allowed range).")
                except ValueError:
                    alert_messages.append(f"{param} value is invalid. Expected a numeric value.")
    else:
        alert_messages.append(f"No parameter details found for machine: {machine}")

    alert_message = " \n\n ".join(alert_messages) if alert_messages else "Your Machine Is In Good Condition: All Given Parameters Are In Machine Acceptable Load/Range"
    return {
        "failure_probability": failure_probability,
        "alert": alert_message
    }