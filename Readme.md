# Brain Tumor Classification — NeuroScan AI

A web app for brain tumor detection and classification using 
Parameter-Efficient Fine-Tuning (PEFT) with VGG16.

## Classes
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** FastAPI, PyTorch
- **Model:** VGG16 with PEFT (BNHead), trained to 99.4% val accuracy

## How to Run

### Backend
```bash
cd backend
pip install fastapi uvicorn torch torchvision h5py pillow python-multipart
uvicorn main:app --reload
```

### Frontend
Open `index.html` with Live Server in VS Code.

## Model
The model file `brain_tumor_vgg16.h5` is not included in this repo 
due to file size. Download it separately and place it in the `backend/` folder.