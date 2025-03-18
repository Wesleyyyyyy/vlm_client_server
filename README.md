# VLM Client-Server

## 🚀 **VLM Client-Server**: A Client-Server Architecture for Vision-Language Model (VLM) Analysis 🌐

Welcome to **VLM Client-Server**! This project sets up a powerful client-server architecture where the server is equipped with a **Vision-Language Model (VLM)**, specifically the **LLaVA 1.5** model. The client can upload images to the server, and in return, the server will provide detailed analysis and captions of the images using the VLM.

## ⚙️ **Key Features**:
- **Client-Server Communication**: Upload images from the client to the server for processing and receive analysis based on the VLM.
- **VLM Analysis**: The server integrates the LLaVA Vision-Language Model, providing intelligent captions and insights from the images.
- **Main Endpoints**:
  - **/chat**: Upload images for VLM-based analysis.
  - ** more endpoints can be integrated

## 🌍 **How It Works**:
1. **Client**: The client uploads images to the server through the provided endpoints.
2. **Server**: The server processes the uploaded images using the LLaVA VLM to generate captions and insights.
3. **Real-time Image Analysis**: The server quickly processes and returns the analysis for the uploaded images.

## 🚀 **Getting Started**:

### 1. Clone the Repository:
```bash
   git clone https://github.com/wenjie-ZH/vlm_client_server.git
   cd vlm-client-server
```

### 2. Install Dependencies:
  Ensure you have Python 3.10+ installed, then install the required dependencies:
```bash
    pip install -r requirements.txt
```

### 3. Run the Server:
Start the FastAPI server using:
```bash
    uvicorn server.py:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 **Technical Stack**:
- **FastAPI and Uvicorn**: A modern, high-performance web framework for building APIs.
- **LLaVA 1.5**: Integrated for intelligent image captioning and analysis.

## 📢 **Contributing**:
We welcome contributions to enhance this project with additional features and endpoints. Feel free to fork the repository and submit pull requests.
